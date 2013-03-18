package com.jd.mltest;

import cern.colt.function.DoubleFunction;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.linalg.Algebra;
import cern.colt.matrix.linalg.SeqBlas;
import cern.jet.math.Functions;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;

import java.util.Random;

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
 * Tikhonov_regularization
 */
public class Glm {

    private final Algebra        algebra        = new Algebra();


    private final DoubleMatrix2D independent;
    private final DoubleMatrix1D dependent;
    private final DoubleMatrix1D thetas;

    //For Feature Scaling.
    private final SummaryStatistics[] columnStats;

    //Size M(num examples). (h)  which is independent * theta
    private final DoubleMatrix1D hypothesies;

    //changes to apply to theta
    private final DoubleMatrix1D deltas;


    //learning rate
    @SuppressWarnings ({"FieldCanBeLocal", "UnusedDeclaration"})
    private final double         alpha;

    //alpha * (1/m) in one or just alpha.
    private final double         modifier;
    private final double         regVal;

    //Are we doing logistic regression or linear regression
    private final boolean        isLogistic;

    //Regularization
    private final Double         lambda;

    private       boolean        isScaled = false;

    /**
     * @param independent
     * @param dependent
     * @param alpha
     * @param isLogistic
     */
    public Glm (DoubleMatrix2D independent,
                DoubleMatrix1D dependent,
                double         alpha,
                boolean        isLogistic,
                Double         lambda ) {


        this.lambda                 = lambda;
        this.isLogistic             = isLogistic;
        this.alpha                  = alpha;

        this.independent            = independent;
        this.dependent              = dependent;

        this.thetas                 = new DenseDoubleMatrix1D(independent.columns());
        this.hypothesies            = new DenseDoubleMatrix1D(dependent.size());

        this.deltas                 = new DenseDoubleMatrix1D(thetas.size());

        Random rand = new Random();
        for( int x=0;x<thetas.size();x++) {
            thetas.set(x,rand.nextGaussian());
        }

        if( this.isLogistic ) {
            //TODO: Not sure if this should be alpha/m or not.
                //I think you are confused between gradient descent and
                //the BFGS, etc that just wants J and derivative.
            this.modifier    = alpha;
        }else {
            this.modifier    = alpha / (double)independent.rows();
        }
        if( lambda != null ) {
            regVal = (alpha*lambda)/dependent.size();
        }else {
            regVal = 0;
        }

        this.columnStats = new SummaryStatistics[thetas.size()];
        for( int x=0;x<columnStats.length;x++) {
            this.columnStats[x] = new SummaryStatistics();
        }

        calculateStats();
    }

    public boolean isRegularized() {
        return( lambda != null );
    }


    private void calculateStats () {
        for( int x=0;x<independent.rows();x++) {
            for( int y=0;y<independent.columns();y++) {
                columnStats[y].addValue(independent.getQuick(x,y));
            }
        }
    }

    /**
     *
     * Scale:   x' = (x - mean) / (max-min)
     * Unscale: x  = (-x' * min) + (x' * max) + mean
     *
     * or
     *
     * Scale:   x' = (x - mean) / stddev
     * Unscale: x  = (x' * stddev) + mean
     *
     * if denom ==0, then skip divide. and reverse..
     *
     */
    public void scaleInputs() {
        if( independent.rows() < 2 ) {
            return;
        }

        for( int x=0;x<independent.rows();x++) {
            //Don't bother to scale the intercept (all 1's)
            for( int y=1;y<independent.columns();y++) {
                double scaled = scale( columnStats[y], independent.getQuick(x,y) );
                independent.setQuick(x,y,scaled);
            }
        }
        isScaled = true;
    }

    private double scale( SummaryStatistics stats, double input ) {
        double mean     = stats.getMean();
        double stddev   = stats.getStandardDeviation();
        double scaled   = (input - mean);

        if( stddev != 0 ) {
            scaled /= stddev;
        }

        return( scaled );
    }

    /**
     * predict using the thetas, and scaling.
     * @param params
     * @return
     */
    public double predict(double... params ) {

        if( isScaled ) {
            //We need to scale our inputs first.
            for( int x=0;x<params.length;x++) {
                //x+1 because we skip the intercept
                params[x] = scale( columnStats[x+1], params[x] );
            }
        }

        double sum = thetas.get(0); //start with intercept
        for( int x=0;x<params.length;x++) {
            sum += params[x] * thetas.get(x+1);
        }

        if( isLogistic ) {
            return( logit(sum) );

        }else {
            //Intercept + x1*theta1 + x2*theta2 ...
            return( sum );
        }
    }


    /**
     * hypothesis is updated with (h - y)
     */
    private void calcHypothesisError() {

        //In Place matrix x vector
        SeqBlas.seqBlas.dgemv(false,1.0,independent,thetas,0,hypothesies);

//        double accum = 0;
//        for( int x=0;x<independent.rows();x++) {
//            accum = 0;
//            for( int y=0;y<independent.columns();y++) {
//                double ind = independent.getQuick(x,y);
//                double theta = thetas.get(y);
//                accum += ind * theta;
//            }
//        }

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
            //(-1/m) * SUM(  (Y * log(h)) + ((1-Y) * log(1-h))  )

            //two vector cross products. gives two scalars. add them. mult by (-1/m)
                //Y * log(h)
                //(1-Y) * log(1-h)


            //TOtally inefficient.. but will figure out the api later...
            DoubleMatrix1D h        = algebra.mult( independent, thetas );
            h.assign(new DoubleFunction() {
                @Override
                public double apply (double val) {
                    return( logit( val ) );
                }
            });

            DoubleMatrix1D lhs      = new DenseDoubleMatrix1D(h.toArray());
            DoubleMatrix1D rhs      = new DenseDoubleMatrix1D(h.toArray());
            DoubleMatrix1D rhDep    = new DenseDoubleMatrix1D(dependent.toArray());

            lhs.assign( Functions.log );
            rhs.assign( new DoubleFunction() {
                @Override
                public double apply (double val) {
                    return( Math.log(1.0-val));
                }
            } );
            rhDep.assign( new DoubleFunction() {
                @Override
                public double apply (double val) {
                    return( 1.0-val);
                }
            } );

            double cost = (-1.0/getNumInstances())*(algebra.mult(dependent,lhs) + algebra.mult(rhDep,rhs));

            if( isRegularized() ) {
                double regularize = 0;
                //(Lambda/(2m))* sum(theta^2)
                //But sum is from 1 to N.. Skip intercept
                for( int x=1;x<thetas.size();x++) {
                    regularize += ( thetas.getQuick(x) * thetas.getQuick(x) );
                }
                regularize *= (lambda/(2*getNumInstances()));
                cost += regularize;
            }
            return(cost);
        }else {
            //assumes hypothesis is already h-y
            //(1/2m) * Sum( (h - y)^2 )
            //sum for all m examples

            //new matrix.. sum of squares of values
            double sumSq = algebra.mult(hypothesies,hypothesies);

            //h is
            return( (1.0/(2.0*hypothesies.size())) * sumSq );

            //if regularized.. +  reg*SUM( theta^2 )
            //                      FOR all theta (except intercept)
        }
    }

    public DoubleMatrix1D getGradient() {

        //hypothesis matrix is set to (h-y)
        calcHypothesisError();

        if( isLogistic ) {
            if( isRegularized() ) {

                //For Regularized LR, the first term (intercept) is not regularized.

                //delta is in hypothesis after calcHypothesisError()

                //delta = (1/m) * SUM( delta * xj )

                // deltas matrix becomes SUM( delta * xj )
                SeqBlas.seqBlas.dgemv(true,1.0,independent,hypothesies,0,deltas);

                //scale by (1/m)
                deltas.assign(Functions.mult(1.0/(double)getNumInstances()));

                //Skip the intercept, and add regularization to all others.
                double regularScale = lambda / (double)getNumInstances();
                for( int x=1;x<deltas.size();x++) {
                    //add (lambda/m)*thetaj
                    double adjust = regularScale*thetas.getQuick(x);
                    deltas.setQuick(x, deltas.getQuick(x) + adjust);
                }

                return( deltas );
            }else {
                return( null );
            }
        }else {
            return( null );
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
        if( isRegularized() ) {

            //intercept is like normal.. theta = (thetas - (deltas*modifier))
            //Linear
                //all others theta = (theta*(1- (alpha*lambda/M) ) - (deltas*modifier)) where lambda is regularization
                //Might also be for logistic.

            //Deltas becomes is Hypothesis error * independent
            SeqBlas.seqBlas.dgemv(true,1.0,independent,hypothesies,0,deltas);

            //Theta0 is the usual Theta0 - (Alpha/M)*Delta
            thetas.set(0, thetas.get(0) - (modifier * deltas.get(0)) );

            //Rest are: Theta*( 1- ((Alpha*Lambda)/M)) - (Alpha/M)*Delta
            for( int x=1;x<thetas.size();x++) {
                double theta = thetas.get(x);
                thetas.set(x, (theta * (1.0 - regVal)) - (modifier * deltas.get(0)) );
            }


        }else {
            //Get Deltas
            SeqBlas.seqBlas.dgemv(true,1.0,independent,hypothesies,0,deltas);
            // thetas = thetas - (deltas*modifier)  in one step
            thetas.assign(deltas, Functions.minusMult(modifier));
        }
    }

    /**
     * The Logit function.. actually probably expit?
     * @param val
     * @return
     */
    public static double logit( double val ) {
        return( 1.0 / (1.0 + Math.exp(-val)));
    }

//    /**
//     * M the number of data instances, rows. etc.
//     * @return
//     */
    public long getNumInstances() {
        return( dependent.size() );
    }

    public int getNumParameters() {
        return( thetas.size() );
    }



//
//    /**
//     * (Lambda/2M) SUM( Theta^2 )
//     * @return
//     */
//    public double getRegularizationValue () {
//        return( (lambda / (2.0*dependent.size())) * algebra.mult(thetas,thetas) );
//    }
}
