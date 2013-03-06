package com.jd.mltest;

import cern.colt.function.DoubleFunction;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.linalg.Algebra;
import cern.colt.matrix.linalg.SeqBlas;
import cern.jet.math.Functions;

/**
 * For my personal learning only.. Use a proper library
 * 
 * A Simple GLM class using Linear Descent.
 * For my learning purposes only... use a real library.
 * Sketch of Linear Regression gradient descent
 * We have our test data, Xij, where i is the instance, and j is the parameter
 * Lets do i is a row, and j is a column
 * We have Yi which is the solution to each Xi linear equation.
 * We have THETAj which are the weights for each Xj,
 * X0 and J0 are going to be the intercept terms, so X0 is always 1.
 *
 * Our cost function will be (1/2m) SUM( (Hi - Yi)^2 )  (for all M examples)
 * Making our partial derivative: (1/m) SUM( (Hi - Yi ) Xi )  (for all M examples)
 * Where Hi is the hypothisis evaluated for Xi is what we get by evaluating an Xi with all of our Thetas, and
 * Yi is obviously the expected result.
 * Xi is the vector of all Xj params...
 * So we will update our Thetas using the partial derivative
 *
 * We also multiply the partial by ALPHA, the learning rate.
 *
 */
public class Glm {

    private final Algebra        algebra        = new Algebra();


    private final DoubleMatrix2D independent;
    private final DoubleMatrix1D dependent;
    private final DoubleMatrix1D thetas;

    //Size M(num examples). (h)  which is independent * theta
    private final DoubleMatrix1D hypothesies;

    //changes to apply to theta
    private final DoubleMatrix1D deltas;


    //learning rate
    @SuppressWarnings ({"FieldCanBeLocal", "UnusedDeclaration"})
    private final double         alpha;

    //alpha * (1/m) in one or just alpha.
    private final double         modifier;

    //Are we doing logistic regression or linear regression
    private final boolean        isLogistic;

    /**
     * @param independent
     * @param dependent
     * @param alpha
     * @param isLogistic
     */
    public Glm (DoubleMatrix2D independent,
                DoubleMatrix1D dependent,
                double         alpha,
                boolean        isLogistic ) {


        this.isLogistic             = isLogistic;
        this.alpha                  = alpha;

        this.independent            = independent;
        this.dependent              = dependent;

        this.thetas                 = new DenseDoubleMatrix1D(independent.columns());
        this.hypothesies            = new DenseDoubleMatrix1D(dependent.size());

        this.deltas                 = new DenseDoubleMatrix1D(thetas.size());

        for( int x=0;x<thetas.size();x++) {
            thetas.set(x,1);
        }

        if( this.isLogistic ) {
            this.modifier    = alpha;
        }else {
            this.modifier    = alpha / (double)independent.rows();
        }
    }

    /**
     * (h - y)
     */
    private void calcHypothesisError() {

        //In Place matrix x vector
        SeqBlas.seqBlas.dgemv(false,1.0,independent,thetas,0,hypothesies);

        //hypothesies = algebra.mult( independent, thetas );

        if( isLogistic ) {
            hypothesies.assign(new DoubleFunction() {
                @Override
                public double apply (double val) {
                    return( logit( val ) );
                }
            });
        }

        hypothesies.assign(dependent, Functions.minus);
    }

    /**
     * Return the current cost function
     * @return
     */
    public double getCost() {
        if( isLogistic ) {

            //Sum for all samples M
            //(-1/m) * SUM(  (Y * ln(h)) + ((1-Y) * ln(1-h))  )

            //two cross products. gives two scalars. add them. mult by (-1/m)
                //Y * ln(h)
                //(1-Y) * ln(1-h)


            return(0);
        }else {
            //(1/2m) * Sum( (h - y)^2 )
            //sum for all m examples

            //new matrix.. sum of squares of values
            double sumSq = algebra.mult(hypothesies,hypothesies);

            //h is
            return( (1.0/(2.0*hypothesies.size())) * sumSq );
        }
    }

    /**
     * Get the Current Theta parameters
     * @return
     */
    public DoubleMatrix1D getThetas () {
        return thetas;
    }

    /**
     * Take one step in the Linear Descent
     */
    public void step() {
        calcHypothesisError();

        //In Place matrix(T) x vector
        SeqBlas.seqBlas.dgemv(true,1.0,independent,hypothesies,0,deltas);

        // thetas = thetas - (deltas*modifier)  in one step
        thetas.assign(deltas, Functions.minusMult(modifier));
    }

    /**
     * The Logit function.. actually probably expit?
     * @param val
     * @return
     */
    public static double logit( double val ) {
        return( 1.0 / (1.0 + Math.exp(-val)));
    }

}
