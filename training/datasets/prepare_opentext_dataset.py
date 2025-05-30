import fire
import json
import os
import random
import math

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer

from transformer_lens.utils import tokenize_and_concatenate


class OpenWebTextFormatter:
    """Classe para formatar o dataset OpenWebText2."""
    
    def format_text(self, input):
        """Formatar texto do OpenWebText2."""
        return {"text": input["text"]}


def prepare_openwebtext2_dataset(
    model_path: str = "Qwen/Qwen3-0.6B",
    hf_user: str = "Gubrg",
    num_tokens: int = 800_000_000,
    context_size: int = 1024,
    hf_token: str | None = None,
    private: bool = False,
    streaming: bool = False
):
    """Gerar dataset tokenizado OpenWebText2 e enviar para HuggingFace."""
    
    formatter = OpenWebTextFormatter()
    
    # Carregar OpenWebText2
    print(">>> Carregando OpenWebText2...")
    dataset = load_dataset("segyges/OpenWebText2", 
                          split="train", 
                          streaming=streaming,
                          token=hf_token)
    
    # Aplicar formatação (mantém o texto como está)
    if not streaming:
        dataset = dataset.map(formatter.format_text).shuffle(seed=42)
    else:
        dataset = dataset.map(formatter.format_text)
    
    # Configurar tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, 
                                              trust_remote_code=True, 
                                              token=hf_token)
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    # Tokenizar dataset
    print(">>> Tokenizando dataset...")
    token_dataset = tokenize_and_concatenate(
        dataset=dataset,
        tokenizer=tokenizer,
        streaming=streaming,
        max_length=context_size,
        column_name="text",
        add_bos_token=False
    )

    if not streaming:
        # Selecionar número de amostras baseado no número de tokens desejado
        num_samples = min(math.ceil(num_tokens / context_size), len(token_dataset))
        token_dataset = token_dataset.select(random.sample(range(len(token_dataset)), num_samples))
        print(f">>> Tokens no dataset = {len(token_dataset) * context_size}")
    else:
        # Para streaming, tomar apenas o número necessário de amostras
        num_samples = math.ceil(num_tokens / context_size)
        token_dataset = token_dataset.take(num_samples)
        print(f">>> Tokens estimados no dataset = {num_samples * context_size}")

    # Enviar para HuggingFace Hub
    repo_id = os.path.join(hf_user, f"{os.path.basename(model_path)}-OpenWebText2-tokenized")
    
    if not streaming:
        token_dataset_dict = DatasetDict({"train": token_dataset})
        token_dataset_dict.push_to_hub(repo_id, token=hf_token, private=private)
    else:
        # Para streaming, precisamos converter para dataset normal primeiro
        print(">>> Convertendo streaming dataset...")
        token_list = list(token_dataset)
        from datasets import Dataset
        token_dataset_normal = Dataset.from_list(token_list)
        token_dataset_dict = DatasetDict({"train": token_dataset_normal})
        token_dataset_dict.push_to_hub(repo_id, token=hf_token, private=private)


if __name__ == "__main__":
    fire.Fire(prepare_openwebtext2_dataset)