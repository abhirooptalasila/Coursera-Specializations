# Structuring Machine Learning Projects


## Single number evaluation metric

The definition of precision is, of the examples that your classifier recognizes as cats, What percentage actually are cats? 
So if classifier A has 95% precision, this means that when classifier A says  something is a cat, there's a 95% chance it really is a cat. And recall is, of all the images that really are cats, what percentage were correctly recognized by your classifier? So what percentage of actual cats, Are correctly recognized? 

In the machine learning literature, the standard way to combine precision and 
recall is something called an F1 score. And the details of F1 score aren't too important, but informally, you can think of this as the average of precision, P, and recall, R. Formally, the F1 score is defined by this formula, it's 2/ 1/P + 1/R.  And in mathematics, this function is called the harmonic mean of precision P and recall R. But less formally, you can think of this as some way that averages precision and recall. 
And it has some advantages in terms of trading off precision and recall. 
But in this example, you can then see right away that classifier A has a better F1 score. And assuming F1 score is a reasonable way to combine precision and recall, 
you can then quickly select classifier A over classifier B. 

So what was seen in this video is that having a single number evaluation metric 
can really improve your efficiency or the efficiency of your team in making those decisions. 

## Satisficing and Optimizing metric

So one thing you could do is combine accuracy and running time into an overall evaluation metric. And so the costs such as maybe the overall cost is accuracy minus 0.5 times running time. But maybe it seems a bit artificial to combine 
accuracy and running time using a formula like this, like a linear weighted sum of these two things. 

Something else you could do instead which is that you might want to choose a classifier that maximizes accuracy but subject to that the running time,  that is the time it takes to classify an image, that that has to be less than or equal to 100 milliseconds. So in this case we would say that accuracy is an optimizing metric because you want to maximize accuracy. You want to do as well as possible on accuracy but that running time is what we call a satisficing metric. Meaning that it just has to be good enough,  it just needs to be less than 100 milliseconds and beyond that you don't really care.

So more generally, if you have N matrix that you care  about it's sometimes reasonable to pick one of them to be optimizing. So you want to do as well as is possible on that one.  And then N minus 1 to be satisficing, meaning that so long as they reach some threshold such as running times faster than 100 milliseconds, 
but so long as they reach some threshold, you don't care how much better it is in that threshold, but they have to reach that threshold. 

So you might care about the accuracy of your trigger word detection system. So when someone says one of these trigger words, how likely are you to actually wake up your device, and you might also care about the number of false positives.  So in this case maybe one reasonable way of combining these two evaluation matrix might be to maximize accuracy, so when someone says one of the trigger words, maximize the chance that your device wakes up. And subject to that, you have at most one false positive every 24 hours of operation.


## Train/dev/test distributions


It turns out, this is a very bad idea because in this example, your dev and test sets come from different distributions. I would, instead, recommend that you find a way to make your dev and test sets come from the same distribution. 

So, what I recommand for setting up a dev set and test set is, choose a dev set and test set to reflect data you expect to get in future and consider important to do well on.  And, in particular, the dev set and the test set here, should come from the same distribution.  So, whatever type of data you expect to get in the future, and once you do well on, try to get data that looks like that. And,  whatever that data is, put it into both your dev set and your test set. 

## Size of the dev and test sets


Set your test set to be big enough to give high confidence in the overall performance of your system. 
Using a test set is always recommended. And the trend has been to use more data for training and less for dev and test, especially when you have a very large data sets. And the rule of thumb is really to try to set the dev set to big enough for its purpose, which helps you evaluate different ideas 

## When to change dev/test sets and metrics


So if you shift Algorithm A the users would see more cat images because you'll see 3 percent error and identify cats, but it also shows the users some pornographic images which is totally unacceptable both for your company, 
as well as for your users. In contrast, Algorithm B has 5 percent error so this  classifies fewer images but it doesn't have pornographic images. 

In this case, the evaluation metric plus the dev set prefers Algorithm A because they're saying, look, Algorithm A has lower error which is the metric you're using but you and your users prefer Algorithm B because it's not letting through pornographic images. So when this happens, when your evaluation metric is no longer correctly rank ordering preferences between algorithms, in this case is mispredicting that Algorithm A is a better algorithm, then that's a sign that you should change your evaluation metric or perhaps your development set or test set. 

follows: this one over m, a number of examples in your development set, 
of sum from i equals 1 to mdev, number of examples in this development set of indicator of whether or not the prediction of example i in your development set is not equal to the actual label i, where they use this notation to denote their predictive value. 

So this way you're giving a much larger weight to examples that are pornographic so that the error term goes up much more if the algorithm makes a mistake on classifying a pornographic image as a cat image. 

So the guideline is, if doing well on your metric and your current dev sets or dev and test sets' distribution, if that does not correspond to doing well on the application you actually care about, then change your metric and your dev test set. 

## Why human-level performance?


And over time, as you keep training the algorithm, maybe bigger and bigger models on more and more data, the performance approaches but never surpasses some theoretical limit, which is called the Bayes optimal error.  So Bayes optimal error, think of this as the best possible error. 

So long as ML is worse than humans, you can:
	Get labeled data from humans
	Gain insight from manual error analysis

## Avoidable bias

sometimes you don't actually want to do too well and knowing what human level performance is, can tell you exactly how well but not too well you want your algorithm to do on the training set. 

You don't really want it to be that much better than 7.5% because you could achieve that only by maybe starting to offer further training so, and instead, there's much more room for improvement in terms of taking this 2% gap and trying to reduce that by using variance reduction techniques such as  regularization or maybe getting more training data. 

call the difference between Bayes error or approximation of Bayes error and the training error to be the avoidable bias. So what you want is maybe keep improving your training performance until you get down to Bayes error but you don't actually want to do better than Bayes error. You can't actually do better than Bayes error unless you're overfitting. 

## Understanding human-level performance


one of the uses of this phrase, human-level error, is that it gives us a way of estimating Bayes error.  What is the best possible error any function could,  either now or in the future

So to recap, having an estimate of human-level performance gives you 
an estimate of Bayes error. And this allows you to more quickly make decisions as to whether you should focus on trying to reduce a bias or trying to reduce the variance of your algorithm.  And these techniques will tend to work well until you surpass human-level performance, whereupon you might no longer have a good estimate of Bayes error that still helps you make this decision really clearly. 

## Surpassing human-level performance

if your error is already better than even a team of humans looking at and discussing and debating the right label, for an example, then it's just also harder to rely on human intuition to tell your algorithm what are ways that your algorithm could still improve the performance? So in this example,  once you've surpassed this 0.5% threshold, your options, your ways of making progress on 
the machine learning problem are just less clear. 

Problems that ML surpasses human-level performance:
	Online advertising
	Product recommendations
	Logistics (predicting transit time)
	Loan approvals

All four of these examples are actually learning from structured data, where you might have a database of what has users clicked on, database of proper support for, databases of how long it takes to get from A to B, database of previous loan applications and their outcomes. And these are not natural perception problems, 
so these are not computer vision, or speech recognition, or natural language processing task. Humans tend to be very good in natural perception task. So it is possible, but it's just a bit harder for computers to surpass human-level  performance on natural perception task. 

All these examples have huge data available. 

## Improving your model performance


Two fundamental assumptions of supervised learning:
	You can fit training set very well
	The training set performance generalizes pretty well to the dev/test set






