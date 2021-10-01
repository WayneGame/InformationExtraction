MAX_LENGTH = 128  # Max Länge des Input Satzes
BATCH_SIZE = 32
EPOCHS = 2 # Bert neigt bei mehreren Epoch zur Überanpassung
BERT_MODEL = "bert-base-german-cased"



# Städte und Länder
SUL = ["Baden-Württemberg","Bayern","Berlin","Brandenburg","Bremen","Hamburg","Hessen","Mecklenburg-Vorpommern","Niedersachsen","Nordrhein-Westfalen","Rheinland-Pfalz","Saarland","Sachsen","Sachsen-Anhalt","Schleswig-Holstein","Thüringen","Schleswig Holstein","Sachsen Anhalt","NRW","Nordrhein Westfalen","Mecklenburg Vorpommern","MeckPomm","MV","MeckPom","NrhW","NW","NrhWf","Baden Württemberg","BW","Berlin","Hamburg","München","Köln","Frankfurt","Stuttgart","Düsseldorf","Leipzig","Dortmund","Essen","Bremen","Dresedn","Hannover","Nürnberg","Duisburg","Bochum","Wuppertal","Bielefeld","Bonn","Münster","Mannheim","Karlsruhe","Augsburg","Wiesbaden","Gelsenkirchen","Braunschweig","Kiel","Chemnitz","Halle","Magdeburg","Freiburg im Breisgau","Krefeld","Mainz","Lübeck","Mainz","Erfurt","Oberhausen","Kassel","Rostock","Hagen","Saarbrücken","Göttingen","Koblenz","Bremerhaven","Bottrop","Jena","Erlangen","Siegen","Gütersloh","Trier","Salzgitter","Remscheid"]

GASTRO = ["Café","Cafe","Eiscafe","Restaurant","Gaststätte","Karaoke","Imbissbude","Imbiss","Imbissstand","Eisdiele","Weinprobe","Eissalon","Bar","Kneipe","Grieche","Italiener","Chinesen","Bistro","Lounge","Fast-Food","Fast Food","Gasthof","Döner","Dönermann","Gasthaus","Landgasthaus","Mc Donalds","Starbucks","Essen gehen","Essen"]

# folgende Sind gekürzt, da das sonst zu viele Trainingssätze für meinen Rechner werden

# Orte einer Stadt
#ORT = ["Fahrschule","Krankenhaus","Klinik","Arzt","Klinikum","Arbeitsamt","Rathaus", "Bank","Bahnhof","Hauptbahnhof","Konsulat","Stadthallen","Altenheim","Friedhof","Kino"]
ORT = ["Fahrschule","Krankenhaus","Klinik","Arzt","Klinikum","Rathaus", "Bank","Bahnhof","Hauptbahnhof","Kino"]

# Irgendwas in der Stadt
#AKTI = ["Einkaufen","Wandern","Städtebummel","bummeln gehen","shoppen","Lebensmittel einkaufen"]
AKTI = ["Einkaufen","Wandern","Städtebummel","bummeln","shoppen"]

# Irgendwas, was mit Größeren Veranstaltungen zu tun hat.
#VER = ["Karneval","Fasching","Weihnachtsmarkt","Demos","Demonstrationen","Christmarkt","Stadtrallye","Glühwein","Feuerwerk","Seifenkistenrennen","Kunstauktion",]
VER = ["Karneval","Fasching","Weihnachtsmarkt","Demos","Demonstrationen"]

#SPORT = ["Turnen","Schlitten fahren","Fitnessstudio","Sommerrodeln","Rodeln","Fußballspiel","Länderspiel","Fußball","Klettern","Kanufahrt","Jagen","Schwimmen","Tauchen","Training","Radfahren","Basketball","Volleyball","Fußball","Bowlen","Boxen","Kampfsport","Football","Cheerleading","CrossFit","Dart","Angeln","Fußball","Beach-Volleyball","Bogenschießen","Bootsfahrten","Fitness-Studio","Gymnastik","Laufen","Leichtathletik","Nordic Walking","Schießen","Sommer-Biathlon","Tennis","Turnen","Volleyball","Skaten","Sportvereine","Joggen","Alpenhüttenzauber","Zirkuswelt mit Kunden","Christbaum fällen mit Glühwein","Kühe melken","Bogen schiessen","Reiten","Pferdeparcour","Kartrennen","Motorsportrennen","Fahrsicherheitstraining","Schneeschuhwanderung","Vollmondwanderung","Klettern","Seilbrücke gemeinsam bauen","Team-Olympiade","Skirennen","Eislaufen","Golftag","Schlittenfahrt","Gletschertrekking (Gemeinsam am Seil mit einem Bergführer über eine Gletscherfläche wandern)","Radeln","Bowling","Billiard","Stockkampf","Qi Gong","Abseilen vom Fels, von einer Klippe oder von einem Gebäude (auf jeden Fall gesichert)","Hochseilgarten","Bogenschiesen","Schanzen-Tubing","Skisprung","Schlittenbau","Geschicklichkeitspacour","Baumklettern","Wildwasserschwimmen","Tipibau","Orientalischer Tanz","Zauberer","Portraits von Karikaturist"]
SPORT = ["Turnen","Schlitten fahren","Fitnessstudio","Rodeln","Fußballspiel","Fußball","Klettern","Kanufahrt","Schwimmen","Training"]

#KULTUR = ["Kino","Theater","Museum","Mahnstätten","Gedenkstätten","Stadtmuseum","Landesmuseum","Oper","Gedenkstätten","Erinnerungsorte","Bibliotheken","Konzerthalle","Bibliothek","Orchester","Archive","Lesung mit bekanntem Autor","Musik","Show","Theater","Unternehmenstheater","Kabarett","Stadttour","Museumsbesuch","Festival","Zirkus","Luftartistik","Tanz","Kaspertheater","Gartenschau","Parkbespielung","Disco-Show","Artisten"]
KULTUR = ["Kino","Theater","Museum","Mahnstätten","Gedenkstätten","Oper","Gedenkstätten","Bibliothek","Konzert","Archiv","Festival","Zirkus"]

#REISEN = ["Ausland","Russland","Rußland","Kanada","Vereinigte Staaten von Amerika","China","Brasilien","Australien","Indien","Argentinien","Kasachstan","Sudan","Algerien","Kongo","Grönland","Mexiko","Saudi-Arabien","Indonesien","Libyen","Iran","Mongolei","Peru","Tschad","Niger","Angola","Mali","Südafrika","Kolumbien","Äthiopien","Bolivien","Mauretanien","Ägypten","Tansania, Vereinigte Republik","Nigeria","Venezuela","Namibia","Pakistan","Mosambik","Türkei","Chile","Sambia","Myanmar","Afghanistan","Somalia","Zentralafrika","Ukraine","Botswana","Madagaskar","Kenia","Frankreich","Jemen","Thailand","Nordamerika","Südamerika","Asien","Europa","Afrika","Australien","USA","Amerika","Hotel"]
REISEN = ["Ausland","Russland","Kanada","Amerika","China","Brasilien","Australien","Indien","Argentinien","Peru","Ägypten","Türkei","Chile","Ukraine","Frankreich","Asien","Europa","Afrika","Australien","USA","Hotel"]

REPLACE_ARRAY = [{"indicator": "$SuL", "word_list": SUL},
                {"indicator": "$ORT", "word_list": ORT},
                {"indicator": "$AKTI", "word_list": AKTI},
                {"indicator": "$VER", "word_list": VER},
                {"indicator": "$SPORT", "word_list": SPORT},
                {"indicator": "$GASTRO", "word_list": GASTRO},
                {"indicator": "$KULTUR", "word_list": KULTUR},
                {"indicator": "$REISEN", "word_list": REISEN},
                ]