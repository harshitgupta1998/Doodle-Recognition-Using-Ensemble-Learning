# Doodle_recognition_project

 Doodle Recognition
Submitted in partial fulfilment of the requirement
of the degree of BACHELOR OF ENGINEERING In COMPUTER ENGINEERING
By
Kanishk Bhatia
Pratik Devnani
Harshit Gupta
Guide:
PROF. SHILPA VERMA
(Associate Professor, Department of Computer Engineering, TSEC)

Introduction
Image-based recognition or search has been available for use for a long time and it has various advantages and disadvantages. Although being the standard methods and techniques which give results and get the job done, there has been a shift in the dynamic of recognition domains. More emphasis has been put on individuality and unexplored avenues. Doodle recognition is the result of such renewed enthusiasm to explore various new domains and to perfect them and get them into a format that is usable in real-life situations.
“A doodle is a drawing made while a person's attention is otherwise occupied” is a very general definition which has been thrown around from the past 2 decades or so. A more suitable definition states that “doodles are simple drawings that can have concrete representational meaning or may just be composed of random and abstract lines, which may or may not have a specific meaning.” Doodles have a far deeper meaning and are extremely important to us than previously perceived.
They can carry a great deal of meaning, and for the artist, they can sometimes be the source of  inspiration for serious artworks. It's important though, to remember that a doodle is not a personality test - it's just a doodle; assessments used by psychologists are scientifically and professionally developed and tested. Thinking about a doodles meaning can help you reflect on your feelings and develop creative ideas, but that is all. Doodles can certainly reveal something about a person, but what? Interpreting them is inexact, to say the least. As handmade marks on paper, they have a great deal in common with graphology. However, no graphologist would use them as a sole indicator. Looking at a collection of various doodles would offer the most helpful insight, especially when coupled with other information, such as handwriting analysis. 
A used case of doodle recognition is auto-drawing or auto-completion of drawing. A trained neural network recognizes the sketch and can suggest clip art based on object recognition. This technology can provide suggestions based on what the user fully or partially draws. Roughly drawn sketches can be instantly converted into neat clip art.
The purpose of this proposed project is to create a system that recognizes and classifies doodles with the highest accuracy while considering time factors and accounting for individual traits that a person displays while doodling.

1.2 Aim & Objectives

The aim of this project is 
•	To design and develop a GUI based platform where a user can draw objects and the system can correctly classify the object. 
•	To ensure that our application works for different classes of objects and that it correctly classifies the objects which are drawn by the users. 
•	To display percentage match and closest resemblance in the form of a pie chart. It matches that the Neural Network made before converging on the result. 
•	To keep in mind the possibility of future expansion and development of the project and making it modular and easy to update.
1.3 Scope

The system to be developed will classify a doodle into a certain class of objects. It will accept as its input a drawing from the user and in return, it will classify the doodle and provide additional information regarding the percentage match and closest resembling options as well.
Before beginning to classify doodles, the algorithm needs to be trained. Often this is done by using supervised learning and historical data. On learning from this training process, the model can predict the class of the doodle accurately. These types of algorithms fall under the domain of machine learning. Machine learning is an application of artificial intelligence (AI) that provides systems with the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves.

The process of learning begins with observations or data, such as examples, direct experience, or instruction, to look for patterns in data and make better decisions in the future based on the examples that we provide the model with. The primary aim is to allow the computer to learn automatically without human intervention or assistance and adjust actions accordingly.
Deep Learning is a subset of Machine Learning Algorithms that is particularly good at recognizing patterns but typically requires a colossal amount of data. Deep learning excels in recognizing objects in images as it is implemented using three or more layers of artificial neural networks where each layer is responsible for extracting one or more features of the image.
Neural Network is a computational model that works in a similar way to the neurons in the human brain. Each neuron takes an input, performs some operations then passes the output to the consequent neuron.
In Deep Learning, a Convolutional Neural Network (CNN) is a class of deep neural networks, most commonly applied to analysing visual imagery. Convolutional Neural Networks are state of the art models for Image Classification, Segmentation, Object Detection, and many other image processing tasks. To get started into the field of Image Processing or to improve the prediction accuracy of the custom CNN models, the knowledge of some of the famous CNN architectures will keep us reach the pace in this competitive and ravenous world. [1] 
Data Collection is also a particularly important step in the process of training our model. The quantity and quality of our data dictates how accurate our model is. To prepare our data to suit the model and our needs is essential for the best functioning of our algorithm. We need to articulate our problem and understand what kind of data we need, how much data do we need, contents of data, size of data, etc.
Data collected from various places will not always be similar or following a standard design. Data needs to be structurally similar and should be similar to our data needs. Some redundancies are removed using deleting the entire row of the data, averaging or by using other techniques. However, for the data, we have to create a universal format for all the datasets to follow.


1.	Google Cloud/Data Set:

The dataset being collected is available on the internet which can be accessed through the cloud. Two versions of the data are given. The raw data is the exact input recorded from the user drawing, while the simplified version removes unnecessary points from the vector information.

The data obtained is further segregated for various purposes. The missing and incorrect data is to be excluded from the dataset and pre-processing all of the data to ensure that they do not contain any unnecessary data, which can be removed during the pre-processing and cleaning of data.

Training/Validation Algorithm:

The training and the validation data is used in the machine learning algorithm for the training the classifier mainly to reassign the weights and the bias of the neural networks, or in case of the logistic regression to reassign the bias, or in the multiclass KNNs the median or the mean is to be reassigned.

a.	Convolution Neural Network (CNN): Convolution Neural Network, like neural networks, are made up of neurons with learnable weights and biases. Each neuron receives several inputs, takes a weighted sum over them, passes it through an activation function and responds with an output. The whole network has a loss function and all the tips and tricks that we developed for neural networks still apply on CNN.

Neural networks, as its name suggests, is a machine learning technique which is modelled after the brain structure. It comprises of a network of learning units called neurons. These neurons learn how to convert input signals (e.g. picture of a cat) into corresponding output signals (e.g. the label “cat”), forming the basis of automated recognition.

Let us take the example of automatic image recognition. The process of determining whether a picture contains a cat involves an activation function. If the picture resembles prior cat images the neurons have seen before, the label “cat” would be activated. Hence, the more labelled images the neurons are exposed to, the better it learns how to recognize other unlabelled images. We call this the process of training neurons. There are four layered concepts we should understand in Convolutional Neural Networks:
•	Convolution
•	ReLu
•	Pooling
•	Full Connectedness (Fully Connected Layer)

There can be multiple renditions of the same object and this makes it tricky for the computer to recognize. But the goal is that if the input signal looks like previous images it has seen before, the “image” reference signal will be mixed into, or convolved with, the input signal. The resulting output signal is then passed on to the next layer.

So, the computer understands every pixel. Usually, the white pixels are said to be -1 while the black ones are 1. 
Now if we would just normally search and compare the values between a normal image and another random rendition, we would get a lot of missing pixels. Hence to fix this problem, we take small patches of the pixels called filters and try to match them in the corresponding nearby locations to see if we get a match. By doing this, the Convolutional Neural Network gets a lot better at seeing similarity than directly trying to match the entire image.

Convolution of an image:

Convolution has the nice property of being translational invariant. Intuitively, this means that each convolution filter represents a feature of interest (e.g. pixels in letters) and the Convolutional Neural Network algorithm learns which features comprise the resulting reference (i.e. alphabet).
We have 4 steps for convolution:
•	Line up the feature and the image
•	Multiply each image pixel by corresponding feature pixel
•	Add the values and find the sum
•	Divide the sum by the total number of pixels in the feature

ReLU Layer:

Rectified Linear Unit (ReLU) transform function only activates a node if the input is above a certain quantity, while the input is below zero, the output is zero, but when the input rises above a certain threshold, it has a linear relationship with the dependent variable. 
The main aim is to remove all the negative values from the convolution. All the positive values remain the same, but all the negative values get changed to zero.

Pooling Layer:

In this layer we shrink the image stack into a smaller size. Pooling is done after passing through the activation layer. We do this by implementing the following four steps:
•	Pick a window size (usually 2 or 3)
•	Pick a stride (usually 2)
•	Walk your window across your filtered images
•	From each window, take the maximum value
The procedure is exactly as same as above and we need to repeat that for the entire image.


The performance is evaluated by computing the percentages of the sensitivity (SE), Specificity (SP), and accuracy (AC) and the respective definition is as follows.

SE = TP/ ( TP + FN ) * 100
SP = TN/ ( TN + FN ) * 100

AC= ( TP + TN ) / ( TN + TP + FN + FP ) * 100

Where  TP is the number of True positives,
TN is the number of True Negatives,
FN is the number of False Negatives
FP is the number of False Positives.

Root- Mean Squared Error is calculated for comparing the error generated in each model. The formula for RSME is:

 
Conclusion

Recognition based technology is coming out of its shell and new research and experimentation are being conducted to explore various avenues which were not known to be existing. Doodles are an important part as it has various use cases like:
•	Human analysis
•	Psychological analysis
•	Object detection
•	Auto-Draw
•	Web Search
•	Information retrieval
Developing a system that accurately classifies objects and does so in a reasonable amount of time is an essential step in the right direction to engender advancements and more research in this field.




