import torch
from pathlib import Path
from tokenizers import Tokenizer
from config import get_config, get_weights_file_path
from train import get_model, get_ds, greedy_decode
from dataset import causal_mask

def translate_sentence(sentence: str, model, tokenizer_src, tokenizer_tgt, config, device):
    """
    Translates a single English sentence to Hindi.
    """
    model.eval()
    with torch.no_grad():
        # Pre-process the sentence
        sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
        eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
        pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)

        # Tokenize source sentence
        enc_input_tokens = tokenizer_src.encode(sentence).ids
        enc_num_padding_tokens = config['seq_len'] - len(enc_input_tokens) - 2

        if enc_num_padding_tokens < 0:
            return "Error: Sentence is too long for the model's sequence length."

        # Create encoder input
        encoder_input = torch.cat([
            sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            eos_token,
            torch.tensor([pad_token] * enc_num_padding_tokens, dtype=torch.int64)
        ]).unsqueeze(0).to(device) # (1, seq_len)

        # Create encoder mask
        encoder_mask = (encoder_input != pad_token).unsqueeze(0).unsqueeze(0).int().to(device) # (1, 1, 1, seq_len)

        # Generate translation using greedy decoding
        model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, config['seq_len'], device)

        # Decode the output tokens back to text
        return tokenizer_tgt.decode(model_out.detach().cpu().numpy())

def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load configuration
    config = get_config()

    # Load tokenizers directly from the JSON files (much faster than get_ds)
    print("Loading tokenizers...")
    tokenizer_src = Tokenizer.from_file(config['tokenizer_file'].format(config['lang_src']))
    tokenizer_tgt = Tokenizer.from_file(config['tokenizer_file'].format(config['lang_tgt']))

    # Load the model
    print("Loading model...")
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    # Specify which epoch to load
    # If you trained on Modal, you can download weights using:
    # modal volume get transformer-weights tmodel_0.pt weights/
    model_epoch = "step_710000" 
    model_filename = get_weights_file_path(config, model_epoch)
    
    if not Path(model_filename).exists():
        print(f"Weight file {model_filename} not found.")
        print("Note: If you trained on Modal.com, sync your weights locally using:")
        print(f"  modal volume get transformer-weights tmodel_{model_epoch}.pt weights/")
        return

    print(f"Loading weights from {model_filename}...")
    state = torch.load(model_filename, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    print("Model loaded successfully!")

    # Example sentences to translate
    test_sentences = [
        "how are you today?",
        "this is a great translation model.",
        "i love machine learning."
    ]

    print("\n" + "="*50)
    print("STARTING CUSTOM INFERENCE")
    print("="*50)

    for sentence in test_sentences:
        translation = translate_sentence(sentence, model, tokenizer_src, tokenizer_tgt, config, device)
        print(f"\nSOURCE:      {sentence}")
        print(f"TRANSLATION: {translation}")
        print("-" * 30)

    # Interactive mode
    print("\nYou can now type your own sentences (type 'exit' to quit):")
    while True:
        user_input = input("\nEnglish: ")
        if user_input.lower() == 'exit':
            break
        
        translation = translate_sentence(user_input, model, tokenizer_src, tokenizer_tgt, config, device)
        print(f"Hindi:   {translation}")

if __name__ == "__main__":
    main()
