import re
import os
import pandas as pd

def parse_tt_file(filepath: str, shorten_dialect: bool = False) -> pd.DataFrame:
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()
    if not lines:
        return pd.DataFrame(columns=["English", "Coptic", "Dialect"])
    
    first_line = lines[0]
    dialect_match = re.search(r'language="(.*?)"', first_line)
    dialect = dialect_match.group(1) if dialect_match else None
    if shorten_dialect and dialect:
        dialect = dialect.split()[0]

    text = "".join(lines[1:])
    trans_block_re = re.compile(
        r'<translation[^>]*translation="(.*?)"[^>]*>(.*?)(?=(?:<translation[^>])|$)',
        flags=re.DOTALL
    )

    records = []
    for m in trans_block_re.finditer(text):
        eng_text = m.group(1).strip()
        block = m.group(2)

        norms = re.findall(r'norm="(.*?)"', block, flags=re.DOTALL)
        if not norms:
            norms = re.findall(r'<entity[^>]*text="(.*?)"', block, flags=re.DOTALL)

        coptic_sentence = " ".join(tok.strip() for tok in norms if tok and tok.strip())
        eng_clean = re.sub(r'\[HERE BEGINNETH\]\s*', '', eng_text, flags=re.IGNORECASE).strip()

        records.append({
            "English": eng_clean,
            "Coptic": coptic_sentence,
            "Dialect": dialect
        })

    return pd.DataFrame(records, columns=["English", "Coptic", "Dialect"])


def build_coptic_dataframe(root_dir: str) -> pd.DataFrame:
    all_dfs = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.endswith(".tt"):
                fpath = os.path.join(dirpath, fn)
                try:
                    df = parse_tt_file(fpath, shorten_dialect=True)
                    if not df.empty:
                        all_dfs.append(df)
                except Exception as e:
                    print(f"Error parsing {fpath}: {e}")
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return pd.DataFrame(columns=["English", "Coptic", "Dialect"])


#-----------------------------------------------------------

!git clone https://github.com/CopticScriptorium/corpora.git
root_dir = "/content/corpora"

print(os.listdir(root_dir)[:20])

# Build the big dataframe
df_all = build_coptic_dataframe(root_dir)

# Add length col
df_all["Coptic_len_words"] = df_all["Coptic"].apply(lambda x: len(x.split()))
df_all["English_len_words"] = df_all["English"].apply(lambda x: len(x.split()))

print(df_all.shape)
df_all.head()


