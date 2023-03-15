DEWEY is a Machine Learning tool used to classify Subject terms from a FUll-TEXT document.
GOing to NEED
   SUBJECT_FILE (TSV SKOS/RDF)
   TRAINING_DATA
   VOCAB_ID
   PROJECT_ID

1. Create dewey.toml or dewey.cfg for ANNIF
1.  Start ANNIF Service   :  annif run 
1.  Load  ANNIF Vocabulary : annif load-vocab VOCAB_ID /Users/scherztc/Workspaces/Annif-tutorial/data-sets/stw-zbw)
1. Import a File (PDF/DOC/JPG)
1. Text(filename_input) is extracted from each digitized document after scanning by means of Optical Character Recognition (OCR),
1. Test(filename_input) will be Pre-Processed. Preprocessing allows us to clean unstructured text
11.   Remove OCR noise 
        words that appear frequently, such as definite or indefinite articles.
11.   tokenization, 
 
       sequence of characters and breaking them
       up into tokens—typically words or phrases—by using common boundaries,
       like punctuation, which is then usually discarded during this step,

111.  Create stopwords.txt
       Filtering may be done next to
       remove stop-words, or words that often appear in a text, but may have little
       semantic value. This includes prepositions, articles, and conjunctions,
       like “a,” “an,” “the,” “but,” and so on. A stop-word list might also contain
       other words that appear frequently or words that appear infrequently,
       depending on the task
11.   filtering, Token.filter
11.   lemmatization, and Token.lemma

      involves further morphological analysis, grouping like words together, such
      as mapping verb forms to their infinite tense.
      11.   stemming: Porter Stemmer for English
      The next step is Class Token method stem  Token.stem
      often to stem the remaining tokens, which reduces inflected words to their
      base or root form. For example, “cataloging” and “cataloged” might be
      reduced to the morphological root “catalog.”
 
1. run ruby classify_dewey filename_input filename_output stopwords.txt
1. run ruby train_dewey filename_input
1. Add new fields to model
1. run ruby scholar_dewey

DEWEY is a LCSH Classifier using Supervised Machine Learning and Natural Language Processing to complete the tasks of a human cataloger by automating Metadata

includes an overview of 

preprocessing, feature extraction, modeling, and evaluation.

Title, Author, Advisor, Date (MS Power Automate)



Genre (Type) (Python Sci-kit, TF-IDF)

classification, 
   Does it have a table of contents or introductions?
   Proper Nouns 
   genre and tries to identify geographic settings, characters, and topics.
   Subject Guidelines
   Subject Access
   Are there external sources?
       Dust jackets, 
       prefaces, 
       publisher
       statements, 
       author interviews, and 
       reviews
keyword extraction, 
       from the description field
named-entity recognition : such as people, places, and things, and then classifying those entities into categories, such as “person” or “corporate body.” 

encoding : convert the text to structured data\

feature selection : TF-IDF, or term frequency-inverse document frequency
    term frequency is calculated based on how often the word
    appears in the text, expressed as a ratio of word occurrence over total number
    of words, while inverse document frequency is represented as the ratio of documents that contain the word.

    scikit-learn in Python
design a model : Text Mining Method 
    Supervised : 
       classification; 
       information extraction, 
           including keyword extraction and named-entity recognition; and 
    UnSupervised : 
        Clustering
	Topic Modeling
Output : Type/Genre predictive accuracy of >70% is respectable, 
         Keyword Extractions

Subject Terms (Keyword)

    Rapid Automatic Keyword Extraction (RAKE)
    Python using RAKE and the Natural Language Toolkit (NLTK)
    module, with a character minimum of 5, a phrase maximum of 3 words,
    and a minimum word frequency of 4 words. Matthew Jocker’s stop-word 
    list was not used in this case, because names are one of the aspects of the
    text that we are most interested in when doing keyword extraction (more
    on this below); the SMART stop list was used instead, which contains 571
    words.
    DBPedia Spotlight22 / Aho-Corasick algorithm
clustering
    Weka to test Simple k-Means, one of the
    most popular clustering algorithms. k-Means partitions a set of documents
    into k clusters where each document belongs to the cluster with the nearest
    mean. To start, the algorithm will randomly choose “centroids” for
    each cluster, then iteratively recalculate the position of those centroids
    until either there is no change in position or some set number of iterations
    has been reached. Identifying meaningful relationships is a matter of
    trial and error, adjusting the value of k and then reviewing the composition
    of each cluster.

    MAchine Learning for LanguagE Toolkit, or MALLET, is a Java application
    for classification, information extraction, and topic modeling, and like
    Weka, is free and easy to use.24 MALLET uses an implementation of Latent
    Dirichlet Allocation (LDA)25


Other types of Machine Learning Cases :


How much / How many? - Regression - Supervised Learning
Which class does this belong to? Classification - Supervised Learning
Are there different groups?  Which one does this belong to? Clustering - UnSupervised Learning
Is this weird? Anomaly Detection - UnSupervised Learning
Which option should I choose? Recommendation - UnSupervised Learning

DEWEY may use other AI Machines in Production :

Chat GPT (OpenAI) : https://openai.com/blog/chatgpt
   Github : https://github.com/openai
   Explore : https://github.com/openai/openai-cookbook
   Uses the openai-client: https://github.com/itikhonenko/openai-client
      API KEY
      ORGANIZATION_ID
   Spell Checker
 
   request_body = {
     model: 'text-davinci-edit-001',
     input: 'What day of the wek is it?',
     instruction: 'Fix the spelling mistakes'
   }
   Openai::Client.edits.create(request_body)

   Image Creator

   request_body = {
     prompt: 'A cute baby sea otter',
     n: 1,                  # between 1 and 10
     size: '1024x1024',     # 256x256, 512x512, or 1024x1024
     response_format: 'url' # url or b64_json
   }
      
   response = Openai::Client.images.create(request_body)

   Connect in Ruby

   require 'openai-client'

   Openai::Client.configure do |c|
     c.access_token    = 'access_token'
     c.organization_id = 'organization_id' # optional
   end

   Find Engine

   Openai::Client.models.find(‘babbage’)
   Openai::Client.models.find(‘davinci’)

   Build Request Body

   request_body = {
     prompt: 'high swim banquet',
     n: 1,                  # between 1 and 10
     size: '1024x1024',     # 256x256, 512x512, or 1024x1024
     response_format: 'url' # url or b64_json
   }


  
Dall-E (OpenAI) : https://openai.com/product/dall-e-2
Stable Diffusion (Stability) : https://stablediffusionweb.com/
Watson (IBM) : https://www.ibm.com/products/watson-explorer 
   Chess
   Content Hub. IBM Watson can propose relevant tags based on content.
Bert (Google)
  Google blog post about BERT,18 an ML technique for NLP, the benefit shown was simply the ability to link a preposition with a noun. 
Aladin (BlackRock)
Mindjourney (MindJourney) : https://www.midjourney.com/home/?callbackUrl=%2Fapp%2F
Kaaros
Tensor Flow (Google)
ANNIF : https://annif.org/
   Command Line Interface, Web Interface, REST API Service Interface
   Install :
   Tutorial :
   Use Case : https://liberquarterly.eu/article/view/10732
   Steps in ANNIF Subject tagging process
      Install
      TFIDF
      Web UI
         Evaluate
             MLLM
                 Hogwarts/FastText
                 Languages / FIltering
	         Ensemble
			NNEnsemble
			Custom
			DVC
             OmiKuji
         REST API

   Project project.toml or project.cfg  ANNIF_PROJECT --projects
    Choose a Subject Vocabulary
            yso-nlf     stw-zbw * (URI)
 	Prepare a Corpus from Training Data
             Load the Vocabulary and Train a Model
                    Suggest subjects for new documents
    PROJECT (VOCABULARY AND LANGUAGE).
	•	YSO NN ensemble English
	•	YSO MLM English
	•	YSO Omikuji Bonsai English
	•	YSO fasttext English
    
IRIS : https://iris.ai/      
PubMED : https://pubmed.ncbi.nlm.nih.gov/30153250/
Other :https://www.aiforlibrarians.com/ai-cases/
Science Direct : Weed Collections : https://www.sciencedirect.com/science/article/pii/S0099133317304160?via%3Dihub
Apache Mahout : https://mahout.apache.org//
Spark MLlib Apache : https://spark.apache.org/mllib/
Stanford : https://library.stanford.edu/blogs/stanford-libraries-blog/2022/07/working-students-library-collections-data
M-Files. Smart subjects provide tag suggestions based on document content
Magellan's AI capabilities include speech and text analytics from contextual hypothesis and meaning deduction.
AWS Innovate: Data and AI/ML Edition

Microsoft PowerAutomation
   How to : https://learn.microsoft.com/en-us/microsoft-365/community/machine-learning-and-managed-metadata
   Application : powerautomate.com
   Models    
     Potential Fields
     PowerBuilder to build and train your model. 
	Structured or Unstructured
     Test
    Flows 
     Triggers
     AI Credits
     Feedback Flows
    Use Cases
         ETD/Student Works : Extract fields from PDF (Title, Author, Advisor, Date, Geo_Subject, Subject) 

Types of Machine Learning  models (use of a computer to follow a pattern)

Programming : Algorithm + Input => Answers
Supervised Learning :  Answer + Input = > Algorithm
Feedback Loop : Re-perturbed feeds classification and classification feed perturbed
Decision Tree : mathematically produced else
Neural Network : binary tree diagrams
Natural Language Processing (NLP_ :  the identification of patterns in spoken or written text.
    Read Understand Derive meaning from Human Languages
    Lanaguage Structures
    INteract Transfer Data
    Feed Document -> encode -> segmentation into sentences by punctuations.  words in the sentences into constiutainet words into tokens.  tokenize.   remove no$
    ALgorithm

    Explain (skip, skipping skipped _ same stemming.
    limitization lemmatizaition
    verbs particle - speech tagging
    Pop culture references movies places news locations- named entitity tagging

    base words tags

    Soccer was invented in Germany.   - >  naive bayes - > Sentiment and Speech

    Segmentation into sentences and store them.
    Tokenizing into words and store them.
    Remove Stop Words (Non Essential Words.)  like are, and, the from sentence segments.
    Stemming treats skip, skipping, skipped as the same words.
    Lemmetization Am Are Is for all genders ich bin, du bist, er/sie/es ist base word (lemma)= be
    Named Entity Tagging of Proper Nouns
    Sentiment and Speech with naive bayes

Reinforcment Learning : data - > model
Cumulative Selection : Building off the last step.  Not starting over everyt ime
Semi-Supervisied Machine Learning : Supervised means there is some human involvement in setting up the tool,
Inductive Reasoning
    The corpus-based approach using a training set, as described above, uses the process of inductive reasoning. This is the kind of thinking that states ‘the sun rose yesterday, the sun rose today, so the chances are the sun will rise tomorrow’. Now, philosophers will argue that inductive reasoning is not scientific. Just because the sun rose yesterday does not mean the sun will rise tomorrow. 


AI Tools
   A Corpus : All text documents in Scholar
   A Training Set :  is a subset of the corpus, which has been tagged in some way to identify the characteristic you are looking for
   A Test Set : collection of documents to be used for trialling the algorithm, to see how successfully it carries out the operation.
       Example : Modified National Institute of Standards and Technology (MNIST) database of handwritten numbers,10
   An Algorithm : The ‘algorithm’ is simply the tool that looks at each item in the corpus and enables a decision to be made. An algorithm may be as simple (and frequently is as simple) as matching a pattern. selecting and applying an algorithm or method

Process Cycle
   Identify
   Explore / Analyze / Encode (Change Everything into a number)
   Model
   Predict
     Clarity (Makes Sense)
     Original (Novelty)
     Useful





Tennis dataset

When do you play tennis? (Class, Outlook, Temp, Windy)

label (y) = things you want to predict
    play / no play

features = 
    outlook, temp, windy

values (x) = { Sunny, Low, Yes }

yak shaving (Remove Turtle Cloud, Kindof, 28%)

Model  

                 Outlook
             Sunny  Overcast   Rainy
           Temp    True          Windy
         low high             False  True
        True  False

How do you know how well it is doing?

Train        - > Test
Use ~ 80%        Use ~ 20%


Train    -> Dev   -> Test
Use ~ 80    Use ~10   Use~10


The Confusion Matrix

valeus (x) = Truth {true, false}
values (y) = Guess {positive, negative}

true positive      false positive

false negative     true negative

precision  = tp / tp + tp
accuracy = tp + tn / tp + tn + fp + fn
recall  = tp / tp + fn


Resources
https://www.youtube.com/watch?v=awGJkRe9m50



NLP:

Feed Document -> encode -> segmentation into sentences by punctuations.  words in the sentences into constiutainet words into tokens.  tokenize.   remove non essential words 9are and the,  stop words
ALgorithm

Explain (skip, skipping skipped _ same stemming.
limitization lemmatizaition
verbs particle - speech tagging
Pop culture references movies places news locations- named entitity tagging

base words tags

Soccer was invented in Germany.   - >  naive bayes - > Sentiment and Speech

Segmentation into sentences and store them.
Tokenizing into words and store them.
Remove Stop Words (Non Essential Words.)  like are, and, the from sentence segments.
Stemming treats skip, skipping, skipped as the same words.
Lemmetization Am Are Is for all genders ich bin, du bist, er/sie/es ist base word (lemma)= be
Named Entity Tagging of Proper Nouns
Sentiment and Speech with naive bayes 





powerapps.microsoft.com
AI Builder
Power Apps
Power Automate.  Do we have this?  All Apps?
Form Processing Model
Choose Model
What information do I want to extract
  Fields
  Tables
  Tables taht span
  Checkboxes

  Order Number. OK
  Customer Name. OK
  Fields. OK

  Collection Share similar layout

  Document tagging.

  Model Summary is a simple algorithms that understands the data in your document

  Train Model

  Start Flow by selecting first option


  Orders in PDF
     Crosswalk data from from PDF into another system.

Copyright of AI


https://www.youtube.com/watch?v=J3d6bx3i4l0

