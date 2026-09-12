import torch
from transformers import AutoTokenizer, AutoModel


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_id = "ibm-granite/granite-embedding-english-r2"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    
    

if __name__ == "__main__":
    main()
    