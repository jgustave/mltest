The goal of this project is to learn more about Machine Learning techniques.
There are many fine ML kits (Like WEKA), but this is not one of them.
This is simply to familiarize myself with the concepts and the implementations of various algorithms.

My First goal will be to implement Batch Linear Regression and Logistic Regression using gradient descent using Colt as
the my linear algebra library.

3/14/13  Success in my initial Linear and Logistic implementaitons using Gradient Descent.
I'm going to now use Mallet to implement the same using gradient and cost functions and L-BFGS...
and finally finishing on Neural Nets.

3/22/13  Success this week getting LBFGS working. Mallet seems to be maximizing, so I had to flip the signs. LBFGS-B (some fortran code)
Seems to be performing the best. No strange errors, minimizes by default, and it converges to an answer much closer to
 a super fine gradient descent.. So Mucho Success! I think next week I will start focusing on getting a NN to work
 using LBFGS-B.  Finally starting to be able to read all those Summations, etc. Lets see how I do when it's a Tripple Sum! :)

3/26/13 Add MiniBatch to the list: http://www.reddit.com/r/MachineLearning/comments/1aycy1/i_cant_get_minibatch_gradient_descent_to_work/
And since this has become a blog of sorts.. Here is Another ultimate goal: Asynchronous gradient descent [http://ai.stanford.edu/~quocle/]

12/9/13 9 months and this project has long since fallen off the list of things I have time for. Maybe more time soon. :)

3/15/14 Hey look at that.. I got to use this for my real job. I added a SimpleLogistic, which is actually reasonably
memory efficient and gets rid of some matrx nonsense from the original version. I've stripped out the regularization for
now. The use case was to calculate a ton of univariate Logistic regressions. I will eventually go back and see about
changing the dependent to something more compact than a matrix of doubles (seems a waste to take 8 bytes to say 1 or 0).
Also want to add AUC/AIC/etc.
