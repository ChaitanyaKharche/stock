import os, textwrap
USE=os.getenv("LLM_PROVIDER","local")
if USE=="local":
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    cfg=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
    tok=AutoTokenizer.from_pretrained("cxllin/Llama2-7b-Finance", trust_remote_code=True)
    model=AutoModelForCausalLM.from_pretrained("cxllin/Llama2-7b-Finance", device_map="auto",
                                              quantization_config=cfg, trust_remote_code=True)
    def _run(prompt):
        ids=tok(prompt,return_tensors="pt").to(model.device)
        out=model.generate(**ids,max_new_tokens=8)[0]
        return tok.decode(out,skip_special_tokens=True)
else:
    from openai import OpenAI; _client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    def _run(prompt):
        r=_client.chat.completions.create(model="gpt-4o",
            messages=[{"role":"system","content":"One-word answer: UP,DOWN or FLAT."},
                      {"role":"user","content":prompt}],temperature=0.1,max_tokens=5)
        return r.choices[0].message.content

def predict(sym, close, ind, news_s, red_s, news_n, red_n):
    p=textwrap.dedent(f"""
    close={close:.2f}
    TA: {', '.join(f"{k}:{v:.2f}" for k,v in ind.items())}
    news_avg={news_s:.2f} ({news_n}), reddit_avg={red_s:.2f} ({red_n})
    """).strip()
    word=_run(p).upper().strip()
    for k in ("UP","DOWN","FLAT"):
        if k in word: return k
    return None
