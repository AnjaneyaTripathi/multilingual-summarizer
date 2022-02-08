from keybert import KeyBERT

doc = """
         Supervised learning is the machine learning task of learning a function that
         maps an input to an output based on example input-output pairs. It infers a
         function from labeled training data consisting of a set of training examples.
         In supervised learning, each example is a pair consisting of an input object
         (typically a vector) and a desired output value (also called the supervisory signal). 
         A supervised learning algorithm analyzes the training data and produces an inferred function, 
         which can be used for mapping new examples. An optimal scenario will allow for the 
         algorithm to correctly determine the class labels for unseen instances. This requires 
         the learning algorithm to generalize from the training data to unseen situations in a 
         'reasonable' way (see inductive bias).
      """

doc_hi = """
         पर्यवेक्षित शिक्षण एक फ़ंक्शन सीखने का मशीन सीखने का कार्य है जो
         उदाहरण के इनपुट-आउटपुट जोड़े के आधार पर आउटपुट में इनपुट मैप करता है। यह अनुमान लगाता है
         प्रशिक्षण उदाहरणों के एक सेट से युक्त लेबल प्रशिक्षण डेटा से कार्य करता है।
         पर्यवेक्षित शिक्षण में, प्रत्येक उदाहरण एक जोड़ी है जिसमें एक इनपुट ऑब्जेक्ट होता है
         (आमतौर पर एक वेक्टर) और एक वांछित आउटपुट मान (जिसे पर्यवेक्षी संकेत भी कहा जाता है)।
         एक पर्यवेक्षित शिक्षण एल्गोरिथ्म प्रशिक्षण डेटा का विश्लेषण करता है और एक अनुमानित कार्य उत्पन्न करता है,
         जिसका उपयोग नए उदाहरणों के मानचित्रण के लिए किया जा सकता है। एक इष्टतम परिदृश्य के लिए अनुमति देगा
         अनदेखी उदाहरणों के लिए कक्षा लेबल को सही ढंग से निर्धारित करने के लिए एल्गोरिदम। ये आवश्यक
         प्रशिक्षण डेटा से अनदेखी स्थितियों में सामान्यीकृत करने के लिए सीखने का एल्गोरिदम a
         'उचित' तरीका (आगमनात्मक पूर्वाग्रह देखें)।
"""

kw_model = KeyBERT()
print(kw_model.extract_keywords(doc_hi, keyphrase_ngram_range=(1, 3), use_maxsum=True))
print(kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 3), use_maxsum=True))