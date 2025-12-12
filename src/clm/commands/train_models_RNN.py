import logging
import os
import os.path
import torch
from pathlib import Path
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from rdkit import rdBase
from clm.datasets import Vocabulary
from clm.models import RNN, ConditionalRNN
from clm.loggers import EarlyStopping, track_loss, print_update
from clm.functions import write_smiles, load_dataset

# suppress Chem.MolFromSmiles error output
rdBase.DisableLog("rdApp.error")
logger = logging.getLogger(__name__)


def add_args(parser):
    parser.add_argument(
        "--representation",
        type=str,
        default="SMILES",
        help="Molecular representation format (one of: SMILES/SELFIES)",
    )

    parser.add_argument(
        "--rnn_type", type=str, help="Type of RNN used (e.g., LSTM, GRU)"
    )

    parser.add_argument(
        "--embedding_size", type=int, help="Size of the embedding layer"
    )

    parser.add_argument("--hidden_size", type=int, help="Size of the hidden layers")

    parser.add_argument("--n_layers", type=int, help="Number of layers in the RNN")

    parser.add_argument("--dropout", type=float, help="Dropout rate for the RNN")

    parser.add_argument("--batch_size", type=int, help="Batch size for training")

    parser.add_argument(
        "--learning_rate", type=float, help="Learning rate for the optimizer"
    )

    parser.add_argument(
        "--max_epochs", type=int, help="Maximum number of epochs for training"
    )

    parser.add_argument("--patience", type=int, help="Patience for early stopping")

    parser.add_argument(
        "--log_every_steps", type=int, help="Logging frequency in steps"
    )

    parser.add_argument(
        "--log_every_epochs", type=int, help="Logging frequency in epochs"
    )

    parser.add_argument(
        "--sample_mols", type=int, help="Number of molecules to sample for evaluation"
    )

    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input file path for training data",
    )

    parser.add_argument(
        "--vocab_file",
        type=str,
        required=True,
        help="Output path for the vocabulary file ({fold} is populated automatically)",
    )

    parser.add_argument(
        "--smiles_file",
        type=str,
        default=None,
        help="File path for additional SMILES data (optional)",
    )

    parser.add_argument(
        "--model_file", type=str, help="File path to save the trained model"
    )

    parser.add_argument(
        "--loss_file", type=str, help="File path to save the training loss data"
    )
    parser.add_argument(
        "--init_model_file",
        type=str,
        default=None,
        help="Optional checkpoint to warm start training from",
    )

    parser.add_argument(
        "--conditional", action="store_true", help="Activate Conditional RNN model"
    )
    parser.add_argument(
        "--conditional_emb",
        action="store_true",
        help="Add descriptor with the input smiles without passing it through an embedding layer",
    )
    parser.add_argument(
        "--conditional_emb_l",
        action="store_true",
        help="Pass the descriptors through an embedding layer and add descriptor with the input smiles",
    )
    parser.add_argument(
        "--conditional_dec",
        action="store_true",
        help="Add descriptor with the rnn output without passing it through decoder layer",
    )
    parser.add_argument(
        "--conditional_dec_l",
        action="store_true",
        help="Pass the descriptors through a decoder layer and add descriptor with the rnn output",
    )
    parser.add_argument(
        "--conditional_h",
        action="store_true",
        help="Add descriptor in hidden and cell state",
    )
    return parser


def _read_vocab_tokens(path):
    if path is None or not os.path.exists(path):
        return []
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]
    

def _write_vocab_tokens(tokens, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for t in tokens:
            f.write(f"{t}\n")


def _merge_vocab(old_tokens, new_tokens):
    merged = list(old_tokens)
    seen = set(old_tokens)
    for t in new_tokens:
        if t not in seen:
            merged.append(t)
            seen.add(t)
    return merged


def _infer_init_vocab_path(init_model_file):
    try:
        p = Path(init_model_file)
        name = p.stem
        if name.endswith("_model"):
            name = name[: -len("_model")]
        parts = name.split("_")
        if len(parts) < 4:
            return None
        fold = parts[-2]
        repr_ = parts[-3]
        dataset = "_".join(parts[:-3])
        return str(p.parent.parent / "inputs" / f"train_{dataset}_{repr_}_{fold}.vocabulary")
    except Exception:
        return None
    

def _adapt_state_dict_for_vocab(sd, old_tokens, merged_tokens, init_model_file):
    emb_key, dec_w_key, dec_b_key = "embedding.weight", "decoder.weight", "decoder.bias"
    if not all(k in sd for k in (emb_key, dec_w_key, dec_b_key)):
        return sd
    if len(old_tokens) ==  len(merged_tokens):
        return sd
    token_to_old = {t: i for i, t in enumerate(old_tokens)}
    emb, dec_w, dec_b = sd[emb_key], sd[dec_w_key], sd[dec_b_key]
    new_emb = torch.zeros(len(merged_tokens), emb.size(1))
    new_dec_w = torch.zeros(len(merged_tokens), dec_w.size(1))
    new_dec_b = torch.zeros(len(merged_tokens))
    for i, tok in enumerate(merged_tokens):
        if tok in token_to_old:
            j = token_to_old[tok]
            new_emb[i] = emb[j]
            new_dec_w[i] = dec_w[j]
            new_dec_b[i] = dec_b[j]
        else:
            torch.nn.init.normal_(new_emb[i], mean=0.0, std=0.2)
            torch.nn.init.normal_(new_dec_w[i])
            new_dec_b[i] = 0.0
    logger.info("Adapting checkpoint vocab from %s -> %s tokens (init_model_file=%s)", len(old_tokens), len(merged_tokens), init_model_file)
    sd[emb_key], sd[dec_w_key], sd[dec_b_key] = new_emb, new_dec_w, new_dec_b
    return sd


def training_step(batch, model, optim, dataset, batch_size):
    loss = model.loss(batch)
    optim.zero_grad()
    loss.backward()
    optim.step()
    validation = dataset.get_validation(batch_size)
    validation_loss = model.loss(validation)
    return loss, validation_loss


def sample_and_write_smiles(model, sample_mols, batch_size, smiles_file, dataset=None):
    sampled_smiles = []
    with tqdm(total=sample_mols) as pbar:
        while len(sampled_smiles) < sample_mols:
            descriptors = None
            if dataset is not None:
                _, _, descriptors = dataset.get_validation(batch_size)
                descriptors = descriptors.to(model.device)
            new_smiles = model.sample(
                n_sequences=batch_size, return_smiles=True, descriptors=descriptors
            )
            sampled_smiles.extend(new_smiles)
            pbar.update(len(new_smiles))
    write_smiles(sampled_smiles, smiles_file)


def train_models_RNN(
    representation,
    rnn_type,
    embedding_size,
    hidden_size,
    n_layers,
    dropout,
    batch_size,
    learning_rate,
    max_epochs,
    patience,
    log_every_steps,
    log_every_epochs,
    sample_mols,
    input_file,
    vocab_file,
    smiles_file,
    model_file,
    loss_file,
    init_model_file=None,
    conditional=False,
    conditional_emb=False,
    conditional_emb_l=True,
    conditional_dec=False,
    conditional_dec_l=True,
    conditional_h=False,
):

    os.makedirs(os.path.dirname(os.path.abspath(model_file)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(loss_file)), exist_ok=True)

    dataset = load_dataset(representation, input_file, vocab_file)

    if conditional:
        model = ConditionalRNN(
            dataset.vocabulary,
            rnn_type=rnn_type,
            n_layers=n_layers,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            dropout=dropout,
            num_descriptors=dataset.n_descriptors,
            conditional_emb=conditional_emb,
            conditional_emb_l=conditional_emb_l,
            conditional_dec=conditional_dec,
            conditional_dec_l=conditional_dec_l,
            conditional_h=conditional_h,
        )
    else:
        model = RNN(
            dataset.vocabulary,
            rnn_type=rnn_type,
            n_layers=n_layers,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            dropout=dropout,
        )

    if init_model_file is not None:
        # Try to locate the vocab that was used to train the init checkpoint
        init_vocab_path = _infer_init_vocab_path(init_model_file)
        init_vocab_tokens = _read_vocab_tokens(init_vocab_path)

        # Read the vocab for this run and append any tokens not in the init vocab
        run_vocab_tokens = _read_vocab_tokens(vocab_file)
        merged_tokens = _merge_vocab(init_vocab_tokens, run_vocab_tokens)

        # If merging added tokens, overwrite the run vocab so training uses the union
        if merged_tokens and merged_tokens != run_vocab_tokens:
            logger.info("Writing merged vocabulary with %d tokens to %s", len(merged_tokens), vocab_file)
            _write_vocab_tokens(merged_tokens, vocab_file)

        # Resize embedding/decoder weights to the merged vocab, preserving old rows
        if merged_tokens:
            # Model moves to GPU during construction inside the model class anyways
            state_dict = torch.load(init_model_file, map_location="cpu")
            state_dict = _adapt_state_dict_for_vocab(
                state_dict,
                init_vocab_tokens or merged_tokens,
                merged_tokens,
                init_model_file,
            )
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(torch.load(init_model_file, map_location=model.device))

    logger.info(dataset.vocabulary.dictionary)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate
    )
    optim = Adam(model.parameters(), betas=(0.9, 0.999), eps=1e-08, lr=learning_rate)
    early_stop = EarlyStopping(patience=patience)

    for epoch in range(max_epochs):
        for batch_no, batch in tqdm(enumerate(loader), total=len(loader)):
            loss, validation_loss = training_step(
                batch, model, optim, dataset, batch_size
            )

            loop_count = (epoch * len(loader)) + batch_no + 1
            if loop_count % log_every_steps == 0 or (
                log_every_epochs and (batch_no + 1) == len(loader)
            ):
                track_loss(
                    loss_file,
                    epoch + 1,
                    loop_count,
                    value=[loss.item(), validation_loss.item()],
                )

                print_update(
                    model,
                    epoch,
                    batch_no + 1,
                    loss.item(),
                    validation_loss.item(),
                    dataset=dataset,
                )

            early_stop(validation_loss.item(), model, model_file, loop_count)

            if early_stop.stop:
                logging.info("Early stopping triggered.")
                break

        if early_stop.stop:
            break

    if log_every_epochs or log_every_steps:
        track_loss(
            loss_file,
            epoch=[None],
            batch_no=[early_stop.step_at_best],
            value=[early_stop.best_loss],
            outcome=["training loss"],
        )

    model.load_state_dict(torch.load(model_file))
    model.eval()
    if smiles_file:
        sample_and_write_smiles(model, sample_mols, batch_size, smiles_file, dataset)


def main(args):
    train_models_RNN(
        representation=args.representation,
        rnn_type=args.rnn_type,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        patience=args.patience,
        log_every_steps=args.log_every_steps,
        log_every_epochs=args.log_every_epochs,
        sample_mols=args.sample_mols,
        input_file=args.input_file,
        vocab_file=args.vocab_file,
        smiles_file=args.smiles_file,
        model_file=args.model_file,
        loss_file=args.loss_file,
        init_model_file=args.init_model_file,
        conditional=args.conditional,
        conditional_emb=args.conditional_emb,
        conditional_emb_l=args.conditional_emb_l,
        conditional_dec=args.conditional_dec,
        conditional_dec_l=args.conditional_dec_l,
        conditional_h=args.conditional_h,
    )
