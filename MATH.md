# DEWEY can do MATH

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

## Interfaces
-   CLI 
-   GUI
-   API


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
1.  Atlassian Rovo - https://www.atlassian.com/blog/announcements/introducing-atlassian-rovo-ai
1.  PhotoMath
1.  Mathway
1.  MathGPTPro
1.  Articulate
1.  BrightBytes
1.  Kahoot!
1.  edFinity
1.  ALICE
1.  LEAN - Decision Tree to prove a Theorem - Proof by AI
1.  NAIRR

# References


# Regulation

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
