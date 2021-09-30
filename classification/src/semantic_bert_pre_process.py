import semantic_bert_config as config
import pandas as pd
from tqdm import tqdm

"""
Die Sätze enthalten unter anderem Platzhalter für Beispielsweise Länder oder Sportarten,
um den Datensatz leicht zu verädnern und deutlich zu vergrößern.
Für jeden Platzhalter wird ein neuer Satz mit dem neuen Wort gebildet.
(Es tauchen nie zwei Platzhalter in einem Satz auf.)
"""
def augment_data(row):
    data = []

    for dic in config.REPLACE_ARRAY:
        if dic["indicator"] in row["TEXT"]:
            for word in dic["word_list"]:
                s = row["TEXT"].replace(dic["indicator"], word)
                data.append({
                    "TEXT": s,
                    "LABEL1": row["LABEL1"],
                    "LABEL2": row["LABEL2"]
                })
    if not data:
        return row.to_frame().T.fillna('none')
    return pd.DataFrame.from_records(data).fillna('none')


"""
Kombiniert jeden des Fragenkatalogs mit den anderen und 
überprüft auf thematische ähnlichkeit.
Sind die Sätze thematisch ähnlich => Label = entailment
andernfalls => Label = neutral TODO: was ist das richtige Label, eine wirklich contradiction ist es ja nicht.

Beispiele aus Standarddatensätzen:
neutral         "A woman with a green headscarf, blue shirt and a very big grin." , The woman is young.
entailment      "A woman with a green headscarf, blue shirt and a very big grin." , The woman is very happy.
contradiction   "A woman with a green headscarf, blue shirt and a very big grin." , The woman has been shot.
"""
def combine_sentences(df):
    combined_list = []
    
    for i, outer_row in tqdm(df.iterrows(), total=len(df)):
        for j, inner_row in df.iterrows():
            
            # Den gleichen Satz müssen wir nicht miteinander vergleichen
            # TODO: Braucht man Dublikate? Siehe Beispiel def remove_dublicates(), falls ja muss aus dem <= ein == werden
            # Bei == sind es 2 hoch n Kombinationen
            # Bei <= sind es (n über 2) Kombinationen
            if i <= j:
                continue

            # Label1 ist nie none
            label_list_outer = [outer_row["LABEL1"]]
            label_list_inner = [inner_row["LABEL1"]]

            if outer_row["LABEL2"] != "none":
                label_list_outer.append(outer_row["LABEL2"])
            if inner_row["LABEL2"] != "none":
                label_list_inner.append(inner_row["LABEL2"])
            
            sim = "neutral"
            if any(label in label_list_outer for label in label_list_inner):
                sim = "entailment"

            combined_list.append({
                "sentence1": outer_row["TEXT"],
                "sentence2": inner_row["TEXT"],
                "similarity": sim
            })
    return pd.DataFrame.from_records(combined_list)

"""
Da ich mir nicht sicher bin, ob das hin und her einen Unterschied macht, 
hatte ich diese Option testhalber noch offen gehalten.

Beispiel:
               sentence1                sentence2  similarity
0     Auf nach Darmstadt    Auf nach Braunschweig     neutral
1  Auf nach Braunschweig       Auf nach Darmstadt     neutral
"""
def remove_dublicates(df):
    df_2 = df
    for i, outer_row in tqdm(df.iterrows(), total=len(df)):
        for j, inner_row in df.iterrows():
            if i == j:
                continue

            if outer_row["sentence1"] == inner_row["sentence2"] and outer_row["sentence2"] == inner_row["sentence1"]:
                df_2.drop(index=i,inplace=True)
                

    return df_2.drop("index", axis="columns")


"""
Kombiniert alle Fragen aus dem Datensatz und weist ein entsprechendes Label zu.
Standard Labels sind: ["contradiction", "entailment", "neutral"]
Auf contradiction wird hier verzichtet.
Das Label wird positiv/entailment, wenn Die Sätze die gleiche Kategorie enthalten

Speichert anschließend die kombinierten und verarbeiteten Daten in einer .csv Datei.
"""
def run():
    # Laden der csv-Datenbank
    df = pd.read_csv("../input/fragen.csv", delimiter=",")
    df.dropna(subset=['TEXT'], inplace=True)

    augmented_df = pd.DataFrame(columns=["TEXT","LABEL1","LABEL2"])
    
    print("\n\nAugmenting Sentences...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        augmented_df = augmented_df.append(augment_data(row), ignore_index = True)
    del(df)

    # durchwürfeln der Daten
    augmented_df = augmented_df.sample(frac=1).reset_index()

    print(augmented_df.head())
    
    # Jetzt haben wir den erweiterten Dataframe 
    # und kombinieren die verschiedenen Sätze miteinander
    print("\n\nCombining Sentences...")
    combined_df = combine_sentences(augmented_df).reset_index()

    del(augmented_df)

    # durchwürfeln der Daten
    combined_df = combined_df.sample(frac=1).reset_index(drop=True)
    
    # Speichern in einer .csv Datei
    print(f"Länge des Datensatzes: {len(combined_df)}")
    combined_df.to_csv("../input/combined_and_labled_questions_shorter.csv", index=False)

if __name__ == "__main__":
    run()