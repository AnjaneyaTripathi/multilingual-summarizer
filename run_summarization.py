# under construction - DO NOT DELETE

from knowledge_graph.kg_summarizer import bart_large_cnn
from knowledge_graph.extractor.extractor import findSVOs
from knowledge_graph.kg_constructor import create_graph
from translation.translation import translate
from enhanced_tfidf.enhanced_tfidf import get_summary
import sys
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt

import en_core_web_lg
nlp = en_core_web_lg.load()


def join_tuple_string(strings_tuple) -> str:
    return ' '.join(strings_tuple)


if __name__ == "__main__":

    topic = sys.argv[1]
    titles = {
        'war': ['Russia Ukraine war', 'रूस यूक्रेन युद्ध', 'रशिया युक्रेन युद्ध'],
        'srilanka': ['Economic meltdown in Sri Lanka', 'श्रीलंका में गहराए आर्थिक संकट', 'श्रीलंकेत आणीबाणी'],
        'will': ['Will Smith slaps Chris Rock', 'विल स्मिथ ने क्रिस रॉक को थप्पड़ जड़ दिया', 'विल स्मिथने ख्रिस रॉकला थप्पड मारली'],
        'imrankhan': ['Imran Khan No-Trust Vote', 'प्रधानमंत्री इमरान खान के खिलाफ अविश्‍वास प्रस्‍ताव', 'इम्रान यांना अविश्वास प्रस्ताव'],
        'bharatpe': ['Ashneer Grover Hits Back At BharatPe CEO For Behen- Tere Bhai Ne..." Post', 'भारतपे के CEO के "बहन-तेरे भाई ने..." वाले पोस्‍ट को लेकर अशनीर ग्रोवर ने किया पलटवार', 'भारतपेचा सह-संस्थापक अश्नीर ग्रोव्हरवर कायदेशीर कारवाई करण्याचा निर्णय, शेअर्सही परत घेणार']
    }
    languages = ['english', 'hindi', 'marathi']
    text = []
    int_summary = []

    with open('./data/original_docs/english/' + topic + '.txt', 'r', encoding="UTF-8") as file:
        text.append(file.read())
    with open('./data/original_docs/hindi/' + topic + '.txt', 'r', encoding="UTF-8") as file:
        text.append(file.read())
    with open('./data/original_docs/marathi/' + topic + '.txt', 'r', encoding="UTF-8") as file:
        text.append(file.read())

    for i in range(0, 3):
        res = get_summary(topic, titles[topic][i], languages[i])
        if(languages[i] != 'english'):
            res = translate(res, languages[i])
            res = res.text
        int_summary.append(res)

    int_summary = ' '.join(int_summary)

    f = sent_tokenize(int_summary)

    nodes = []
    for sentence in f:
        tokens = nlp(sentence)
        svos = findSVOs(tokens)
        nodes.append(svos)

    final_nodes = []

    for node in nodes:
        for j in node:
            if(len(j) == 3):
                final_nodes.append(j)

    result = map(join_tuple_string, final_nodes)
    result = ". ".join(result)

    create_graph(final_nodes)
    plt.savefig('./data/knowledge_graphs/kg_' + topic + '.png')

    final_text = int_summary + result

    final_summary = bart_large_cnn(final_text)
    final_summary = final_summary[0]['summary_text']

    print('---summariztion done:: ', final_summary)
