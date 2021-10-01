import pandas as pd
import sim_tfidf_utils as utils
import sim_tfidf_config as config

from tqdm import tqdm



"""
Behandelt den html Text und gibt einen verarbeiteten String zurück.
"""
def pre_process(url):
    try:
        html = utils.get_webpage(url)
    except:
        return None

    link = utils.get_cleand_url(url)
    body = utils.get_cleaned_webpage_text(html)
    meta = utils.get_tags(html)
    title = utils.get_webpage_title(html)
    h_elements = utils.get_h_elements(html)
    
    # TODO: ist für den BERT corpus test erstmal ausgeklammert
    link = utils.add_keywords(link)
    body = utils.add_keywords(body)
    meta = utils.add_keywords(meta)
    title = utils.add_keywords(title)
    h_elements = utils.add_keywords(h_elements)

    return {
            "url"  : url,
            "link" : link,
            "body" : body,
            "meta" : meta,
            "title" : title,
            "h_elements" : h_elements
        }

"""
Speichert die ausgewählten Teile der als interessant eingestuften Webseiten
in eine .csv Datei.
"""
def save_corpus(df, save_columns=["link", "body", "meta", "title", "h_elements"]):

    ret_df = df[save_columns].agg('. '.join, axis=1).to_frame() # TODO: Das ist nur für BERT
    #ret_df = df[save_columns].agg(' '.join, axis=1).to_frame() # TODO: für nicht BERT
    ret_df.columns = ["corpus"]
    ret_df["url"] = df["url"]

    ret_df.to_csv(f"{config.CORPUS_SAVE_PATH}_bert") # TODO "all" weg und nur config.CORP...

if __name__ == "__main__":
    # Preprocessing der Interessanten Links 
    # Verläuft extern, da verschiedenes PreProcessing ausprobiert werden soll
    # nur interessante
    interesting_webpages_df = pd.read_csv("../input/interesting_webpages.csv")
    # Alles
    #interesting_webpages_df = pd.read_csv("../input/webpages.csv")

    webpages_infos = []

    for idx, row in tqdm(interesting_webpages_df.iterrows(), total=len(interesting_webpages_df)):
        nina = pre_process(row['LINK'])
        
        # Bei manchen Webseiten kommt es zu Fehlern oder TimeOuts, in diesem Fall wird
        # None zurückgegeben und die Seite einfach übersprungen. (Besonders anfällig schien bei den Tests die Webseite Wiesbadens.)
        if nina:
            webpages_infos.append(nina)

    corpus_df = pd.DataFrame.from_records(webpages_infos)

    save_corpus(corpus_df, save_columns=config.CORPUS_COLUMNS)



    # speichern des Vectorizers und des Corpus, sowie der dazugehörigen .csv mit den Links
    # sentence encodings u. vectorizer Count TF/IDF


    # Später dann Laden des Corpus usw.
    