# Theory : Other types of Machine Learning Cases :

The real skill in Machine Learning is determining which questions can be answered by Machine Learning.

1. How much / How many? - Regression - Supervised Learning
1. Which class does this belong to? Classification - Supervised Learning
1. Are there different groups?  Which one does this belong to? Clustering - UnSupervised Learning
1. Is this weird? Anomaly Detection - UnSupervised Learning
1. Which option should I choose? Recommendation - UnSupervised Learning
1. Translation
1. Test Sentiment
1. Summarization
   - Analyze this Graph
1. Parsing
1. Keywords
1. Conversation
1. Text Generation
   - Text Extraction
1. Coding
   - Co-pilot
   - Visualiztions
   - Imaginations

# DEWEY can leverage ANNIF to classify subject terms from a FUll-TEXT document using ANNIF.

## URLS
   https://liberquarterly.eu/article/view/10732

## Interfaces
-   CLI 
-   GUI
-   API

## Prerequisites
-   SUBJECT_FILE (TSV SKOS/RDF)
-   TRAINING_DATA
-   VOCAB_ID : 
    -  https://skos.um.es/unescothes/
    -  https://id.loc.gov/download/
    -  https://github.com/NatLibFi/Annif/wiki/Subject-vocabulary-formats
-   PROJECT_ID

## ANNIF Instructions

1. Git Clone : git@github.com:NatLibFi/Annif.git

Install pipx and Poetry if you don't have them. First pipx:
python3 -m pip install --user pipx
python3 -m pipx ensurepath

1. Open a new shell, and then install Poetry:
pipx install poetry

1. Create a virtual environment and install dependencies:
poetry install --all-extras

1. Enter the virtual environment:
poetry shell

1. You will also need NLTK data files:
python -m nltk.downloader punkt

1. Copy projects.cfg.dist to projects.cfg

1. Load Subject Vocabulary

1. git clone : git@github.com:NatLibFi/Annif-corpora.git
annif load-vocab yso /path/to/Annif-corpora/vocab/yso-skos.ttl

1. We will train the model using the the English language training data generated from Finna.fi metadata:
annif train tfidf-en /path/to/Annif-corpora/training/yso-finna-en.tsv.gz

Other Models to consider MLLM, Hogwarts/Fast, Languages, NNEnsemble, Custom, DVC, OmiKuji

1. Start up the application:
annif run

1. Using Docker
docker run -v /Users/scherztc/Workspaces/dewey/annif-docker:/annif-projects -u $(id -u):$(id -g) -it quay.io/natlibfi/annif bash


1. Preprocessing allows us to clean unstructured text
  1.  Remove OCR noise (works that appear frequently, such as definite and indefinite articles.
  2.  Tokenization (sequencing characters and breaking them up into words, phrases, sentences)
  3.  Create stopwords.txt (This includes prepositions, articles, and conjunctions, like “a,” “an,” “the,” “but,” and so on.) 
  4.  Filtering (Token.filter)
  5.  Lemmatization (Token.lemma : morphological analysis and grouping like words together i.e. verb tenses)     
  6.  Stemming (Token.stem : Reduces inflected words to their base or root form)

1. Test Document (GUI)
- Visit localhost:5000
- Copy/Paste Text
- Select TF-IDF English
- Run

1. or CLI

- cat document.txt | annif suggest tfidf-en

1. or CLI with folder of documents

- annif eval tfidf-en /path/to/documents/

1. Explore API
The Swagger UI documentation for the REST API is at http://localhost:5000/v1/ui/


# DEWEY can leverage PowerAutomation from Microsoft to determine the Title, Author, and Date of works submitted

1. Power Automate (https://powerautomate.microsoft.com/)
   1.  AI Builder
     1.  Models
     1.  Training Document
     1.  Information to Extract
   1. My Flows
     1.  Cloud Flows
         1.  Use form processing to extract data from documents triggered manually.
         1.  Flow Checker Feedback Flows and Run

   1. How to : https://learn.microsoft.com/en-us/microsoft-365/community/machine-learning-and-managed-metadata

# DEWEY can leverage Python, Sci-kit, and TF-IDF to build a model from scratch for determine the Genre (Type) of document is being submitted.

1. Classification, 
   -  Does it have a table of contents or introductions?
   -  Proper Nouns 
   -  genre and tries to identify geographic settings, characters, and topics.
   -  Subject Guidelines
   -  Subject Access
   -  Are there external sources?
       -  Dust jackets, 
       -  prefaces, 
       -  publisher
       -  statements, 
       -  author interviews, and 
       -  reviews
1. Keyword extraction, 
   -  From the description field

1. Named-entity recognition : such as people, places, and things, and then classifying those entities into categories, such as “person” or “corporate body.” 

1. Encoding : convert the text to structured data\
   - Transfomers

1. Feature selection : TF-IDF, or term frequency-inverse document frequency
    term frequency is calculated based on how often the word
    appears in the text, expressed as a ratio of word occurrence over total number
    of words, while inverse document frequency is represented as the ratio of documents that contain the word.

1. Design a model : Text Mining Method  
   -    Supervised : 
   -    classification; 
   -    information extraction, 
        -   including keyword extraction and named-entity recognition; and 
   -    UnSupervised : 
        -  Clustering
	-  Topic Modeling

   -  Output : Type/Genre predictive accuracy of >70% is respectable, 
        -  Keyword Extractions

1.  Subject Terms (Keyword)

    -  Rapid Automatic Keyword Extraction (RAKE)
    -  Python using RAKE and the Natural Language Toolkit (NLTK)
    module, with a character minimum of 5, a phrase maximum of 3 words,
    and a minimum word frequency of 4 words. The SMART stop list was used instead, which contains 571
    words. 
    -  DBPedia Spotlight22 / Aho-Corasick algorithm

1.  Clustering
    -  Weka to test Simple k-Means, one of the
    most popular clustering algorithms. k-Means partitions a set of documents
    into k clusters where each document belongs to the cluster with the nearest
    mean. 
    -  To start, the algorithm will randomly choose “centroids” for
    each cluster, then iteratively recalculate the position of those centroids
    until either there is no change in position or some set number of iterations
    has been reached. 
    -  Identifying meaningful relationships is a matter of
    trial and error, adjusting the value of k and then reviewing the composition
    of each cluster.

# DEWEY can leverage MALLET for classifying documents.
    -  Machine Learning for LanguagE Toolkit, or MALLET, is a Java application
    for classification, information extraction, and topic modeling, and like
    Weka, is free and easy to use.24 
    -  MALLET uses an implementation of Latent
    -  Dirichlet Allocation (LDA)25

# DEWEY can leverage Chat GPT (3.5, 3.5-turbo, 4.0, 4.0-=turbo) as a spell checker, voice response, or Image Creator :

1.   Chat GPT (OpenAI) : https://openai.com/blog/chatgpt
   -  Github : https://github.com/openai
   -  Explore : https://github.com/openai/openai-cookbook
   -  Uses the openai-client: https://github.com/itikhonenko/openai-client
      -  API KEY
      -  ORGANIZATION_ID
   -  Dall-E (OpenAI) : https://openai.com/product/dall-e-2

   1.  Spell Checker
 
         request_body = {
            model: 'text-davinci-edit-001',
            input: 'What day of the wek is it?',
         instruction: 'Fix the spelling mistakes'
         }
         Openai::Client.edits.create(request_body)

   1.  Image Creator

         request_body = {
            prompt: 'A cute baby sea otter',
            n: 1,                  # between 1 and 10
            size: '1024x1024',     # 256x256, 512x512, or 1024x1024
            response_format: 'url' # url or b64_json
         }
      
        response = Openai::Client.images.create(request_body)

   1. Connect in Ruby

        require 'openai-client'

        Openai::Client.configure do |c|
          c.access_token    = 'access_token'
          c.organization_id = 'organization_id' # optional
        end

   1. Find Engine

        Openai::Client.models.find(‘babbage’)
        Openai::Client.models.find(‘davinci’)

   1. Build Request Body

        request_body = {
           prompt: 'high swim banquet',
           n: 1,                  # between 1 and 10
           size: '1024x1024',     # 256x256, 512x512, or 1024x1024
           response_format: 'url' # url or b64_json
        }

    1. Playground interface : https://platform.openai.com/playground?mode=chat

#  DEWEY can leverage Machine Learning and Large Language Models

    1.  Repository : https://huggingface.co/
    1.  The Bloke : https://huggingface.co/TheBloke
    1.  Lone Striker : https://huggingface.co/LoneStriker
    1.  WebGUI : https://github.com/oobabooga/text-generation-webui
    1.  Stable Diffusion : https://github.com/AUTOMATIC1111/stable-diffusion-webui
    1.  Voice Changer : github.com/w-okada/voice-changer
    1.  Real Time Voice : https://github.com/RVC-Project/Retrieval-based-VOice-Conversion-WebUI
    1.  RVC : voice-models.com  and weighs.gg

#  Other AI Engines to Explore
  
1.  Stable Diffusion (Stability) : https://stablediffusionweb.com/  or civitai.com
1.  Watson (IBM) : https://www.ibm.com/products/watson-explorer 
   1. Chess
   1. Content Hub. IBM Watson can propose relevant tags based on content.
1.  Bard/Palm 2 (Google)
   1. Google blog post about BERT,18 an ML technique for NLP, the benefit shown was simply the ability to link a preposition with a noun. 
1.  Aladin (BlackRock)
1.  Mindjourney (MindJourney) : https://www.midjourney.com/home/?callbackUrl=%2Fapp%2F
1.  Kaaros
1.  Tensor Flow (Google)
1.  IRIS : https://iris.ai/      
1.  Claude https://www.anthropic.com/index/claude-2
1.  https://marketplace.atlassian.com/apps/1224655/scrum-maister?hosting=cloud&tab=overview
1.  Bing (free)
1.  Claude 2 (free) by Anthropic
1.  Grok by X (Twitter)
1.  Open-source models (FREE) available on Huggingface https://huggingface.co/
1.  Llama 2 by Meta
1.  Flan, Falcon, Orca, Beluga, Mistral, Mixtral, Phi2
1.  LMStudio (Windows, Mac, Linux) - install and run models
1.  Pinokio.computer browser - install and run models

# References

1.  PubMED : https://pubmed.ncbi.nlm.nih.gov/30153250/
1.  Other :https://www.aiforlibrarians.com/ai-cases/
1.  Science Direct : Weed Collections : https://www.sciencedirect.com/science/article/pii/S0099133317304160?via%3Dihub
1.  Apache Mahout : https://mahout.apache.org//
1.  Spark MLlib Apache : https://spark.apache.org/mllib/
1.  Stanford : https://library.stanford.edu/blogs/stanford-libraries-blog/2022/07/working-students-library-collections-data
1.  M-Files. Smart subjects provide tag suggestions based on document content
1.  Magellan's AI capabilities include speech and text analytics from contextual hypothesis and meaning deduction.
1.  AWS Innovate: Data and AI/ML Edition
1.  Data Science : https://www.dataplusscience.com/GenerativeAI.html
1.  AI Got Talent : https://dataplusscience.com/files/UCCBAGenAI20240206.pdf

# Regulation

MEPs substantially amended the list to include bans on intrusive and discriminatory uses of AI systems such as:


1.  “Real-time” remote biometric identification systems in publicly accessible spaces;
1.  “Post” remote biometric identification systems, with the only exception of law enforcement for the prosecution of serious crimes and only after judicial authorization;
1.  Biometric categorisation systems using sensitive characteristics (e.g. gender, race, ethnicity, citizenship status, religion, political orientation);
1.  Predictive policing systems (based on profiling, location or past criminal behaviour);
1.  Emotion recognition systems in law enforcement, border management, workplace, and educational institutions; and
1.  Indiscriminate scraping of biometric data from social media or CCTV footage to create facial recognition databases (violating human rights and right to privacy).

# AI Feedback Loop

-  Programming : Algorithm + Input => Answers
-  Supervised Learning :  Answer + Input = > Algorithm
-  Feedback Loop : Re-perturbed feeds classification and classification feed perturbed
-  Decision Tree : mathematically produced else
-  Neural Network : binary tree diagrams
-  Natural Language Processing (NLP_ :  the identification of patterns in spoken or written text.
    -  Read Understand Derive meaning from Human Languages
    -  Lanaguage Structures
    -  INteract Transfer Data
    -  Feed Document -> encode -> segmentation into sentences by punctuations.  words in the sentences into constiutainet words into tokens.  tokenize.   remove no$
    -  ALgorithm
    -  Explain (skip, skipping skipped _ same stemming.
    -  limitization lemmatizaition
    -  verbs particle - speech tagging
    -  Pop culture references movies places news locations- named entitity tagging

>    Segmentation into sentences and store them.
>    Tokenizing into words and store them.
>    Remove Stop Words (Non Essential Words.)  like are, and, the from sentence segments.
>    Stemming treats skip, skipping, skipped as the same words.
>    Lemmetization Am Are Is for all genders ich bin, du bist, er/sie/es ist base word (lemma)= be
>    Named Entity Tagging of Proper Nouns
>    Sentiment and Speech with naive bayes

-  Reinforcment Learning : data - > model
-  Cumulative Selection : Building off the last step.  Not starting over everyt ime
-  Semi-Supervisied Machine Learning : Supervised means there is some human involvement in setting up the tool,
-  Inductive Reasoning
   -   The corpus-based approach using a training set, as described above, uses the process of inductive reasoning. This is the kind of thinking that states ‘the sun rose yesterday, the sun rose today, so the chances are the sun will rise tomorrow’. Now, philosophers will argue that inductive reasoning is not scientific. Just because the sun rose yesterday does not mean the sun will rise tomorrow. 

# AI Terms
   - A Corpus : All text documents in Scholar
   - A Training Set :  is a subset of the corpus, which has been tagged in some way to identify the characteristic you are looking for
        Common Crawl 
        RefinedWeb
        The Pile
        C4
        Starcoder
        BookCorpus o ROOTS
        Wikipedia o Red Pajama

   - A Test Set : collection of documents to be used for trialling the algorithm, to see how successfully it carries out the operation.
       - Example : Modified National Institute of Standards and Technology (MNIST) database of handwritten numbers,10
   - An Algorithm : The ‘algorithm’ is simply the tool that looks at each item in the corpus and enables a decision to be made. An algorithm may be as simple (and frequently is as simple) as matching a pattern. selecting and applying an algorithm or method

# AI Process Cycle
   1. Identify
   1. Explore / Analyze / Encode (Change Everything into a number)
   1. Model
   1. Predict
       1. Clarity (Makes Sense)
       1. Original (Novelty)
       1. Useful
   1. Feedback


# Tennis dataset (Machine Learning)

This is a famous dataset used to demonstrate how Machine Learning and Decision Making work.

When do you play tennis? (Class, Outlook, Temp, Windy)

- label (y) = things you want to predict
    - play / no play

- features = 
    - outlook, temp, windy

- values (x) = { Sunny, Low, Yes }

- yak shaving (Remove Turtle Cloud, Kindof, 28%)

## Model  

>                 Outlook
>             Sunny  Overcast   Rainy
>           Temp    True          Windy
>         low high             False  True
>        True  False

How do you know how well it is doing?

> Train        - > Test
> Use ~ 80%        Use ~ 20%


> Train    -> Dev   -> Test
> Use ~ 80    Use ~10   Use~10


## The Confusion Matrix

-  values (x) = Truth {true, false}
-  values (y) = Guess {positive, negative}

>  true positive      false positive

>  false negative     true negative

-  precision  = tp / tp + tp
-  accuracy = tp + tn / tp + tn + fp + fn
-  recall  = tp / tp + fn


Resources
https://www.youtube.com/watch?v=awGJkRe9m50
