# Improved-COMPAS-System

### Background on COMPAS

COMPAS evaluates criminals on over 100 factors and outputs a risk score which indicates how likely it is that someone will recidivate (go on to commit another crime in the future). These scores are then taken into consideration by judges when assigning sentences, determining bail/parole eligibility, etc.

ProPublica reviewed the output of COMPAS on a dataset of over 7000 individuals from Broward County, Florida. They found that the algorithm correctly predicted recidivism at similar rates for both white (59%) and black defendants (63%). However, when the algorithm was incorrect it tended to skew very differently for each of these groups. White defendants who re-offended within two years were mistakenly labelled low risk almost twice as often as their black counterparts. Additionally, black defendants who did not recidivate were rated as high-risk at twice the rate of comparable white defendants.

### Goal 

Create a model to replace COMPAS from 
1) linear support vector machine based regression model
2) A feed forward neural network
3) A naive Bayes classifier to replace COMPAS

Train the model on the past 2 years data collected from Broward County. 

Apply Post Processing Methods that enforce various constraints in attempts to reflect different measures of fairness. These include:
1) Maximum Accuracy 
2) Single threshold
3) Predictive parity
4) Demographic parity
5) Equal Opportunity

Implement all these 5 methods on the above 3 models and then determine which model/fairness combination to put forward as your finished
product to replace existing COMPAS system.

### Proposed Solution 

The biggest issue with COMPAS was the racial disparity on how it mistreated black defendants and handed harsher sentences by labeling them a high risk to recidivism even when they went on to not commit crimes in the future whereas labeled white defendants’ low risks to recidivism when they went on to commit a crime in future. The goal is not to see the white defendants be considered more guilty but to make sure that a person who is capable of change, who is not going to commit a crime in the future, a person deserving a lesser sentence be not flagged wrong, therefore the proposed model is a neural network which uses demographic parity to get the highest accuracy while giving the lowest false positive rate for all races. This makes sure that no innocent man gets a harsher sentence across all races. 

### Why this Model and Post Processing Method?

The proposed solution of using a neural network model with demographic parity for the highest accuracy is a better choice than the other methods since the primary goal is to make sure that False Positive Rate remains lowest. If the FPR is high, that means that a person who will not commit a crime in the future is marked as likely to recidivism. That would mean a harsher sentence for an innocent person; therefore, making sure that the lowest FPR is kept among all models and methods with the choices. It also solves the racial disparity among the different races seen in the COMPAS algorithm as in the new model, FPR remains in around 2 percent difference for all races. In contrast, in the original COMPAS algorithm, Black defendants who do not recidivate were nearly twice as likely to be classified by COMPAS as higher risk than their white counterparts (45 percent vs. 23 percent). So this model has achieved closing the gap as in the current model, the FPR for the African American population is 41 percent, Caucasian is 39 percent, and approximately 40 percent for both Hispanic and other races. The only significant disparity in numbers among races can be seen in FNR and TPR, it can be considered a concern, but according to the criminal law, Blackstone’s ratio says that “It is better that ten guilty persons escape than one innocent suffer.” Therefore, the focus is on keeping the FPR at a minimum to ensure that no innocent suffer even if it increases the FNR.  




