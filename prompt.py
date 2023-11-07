import fire
from llama import Llama
from typing import List

# launch command
# """
# torchrun --nproc_per_node 1 prompt.py \
#     --ckpt_dir llama-2-7b-chat/ \
#     --tokenizer_path tokenizer.model \
#     --max_seq_len 512 --max_batch_size 6
# """

print("종료를 위해서 exit을 입력하세요. ")

def app(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
    ):
    
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
     
    while True:
        value = input("프롬프트 입력 : ")
        if value == 'exit':
            break
        prompts: List[str] = [
            str(value)
        ]
        results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,)
        
        for prompt, result in zip(prompts, results):
            print("response : " + results[0]['generation'])

if __name__ == "__main__":
    fire.Fire(app)
