DEWEY is a Machine Learning tool used to classify Subject terms from a FUll-TEXT document.

DEWEY is a Classifier using Supervised Machine Learning and Natural Language Processing to complete the tasks of a human cataloger.

Other types of Machine Learning Cases :

How much / How many? - Regression - Supervised Learning
Which class does this belong to? Classification - Supervised Learning
Are there different groups?  Which one does this belong to? Clustering - UnSupervised Learning
Is this weird? Anomaly Detection - UnSupervised Learning
Which option should I choose? Recommendation - UnSupervised Learning

DEWEY may use other AI Machines in Production :

Chat GPT (OpenAI)
Watson (IBM
Bard (Google)
Aladin (BlackRock)
Mindjourney
Kaaros
Tensor Flow (Google)


Types of Processing models

Decision Tree mathematically produced else
Neural Network
Natural Language Processing
    Read Understand Derive meaning from Human Languages
    Lanaguage Structures
    INteract Transfer Data
data - > model

Process
   Identify
   Explore / Analyze / Encode
   Model
   Predict

Encode = Change everything into a number


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





Microsoft AI Builder to Extract Data from PDF
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


https://www.youtube.com/watch?v=J3d6bx3i4l0

