package com.jd.mltest;

import cc.mallet.optimize.LimitedMemoryBFGS;
import cc.mallet.optimize.Optimizable;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;

import java.util.Arrays;
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
public class GlmLess {

    private final DoubleMatrix2D independent;
    private final DoubleMatrix1D dependent;
    private final DoubleMatrix1D thetas;

    //For Feature Scaling.
    private final SummaryStatistics[] columnStats;

    //changes to apply to theta
    private final DoubleMatrix1D deltas;


    //learning rate
    @SuppressWarnings ({"FieldCanBeLocal", "UnusedDeclaration"})
    private       double         alpha;

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
    public GlmLess (DoubleMatrix2D independent,
                    DoubleMatrix1D dependent,
                    double alpha,
                    boolean isLogistic,
                    Double lambda) {


        this.lambda                 = lambda;
        this.isLogistic             = isLogistic;
        this.alpha                  = alpha;

        this.independent            = independent;
        this.dependent              = dependent;

        this.thetas                 = new DenseDoubleMatrix1D(independent.columns() + 1);

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

    public void setAlpha (double alpha) {
        this.alpha = alpha;
    }

    public double getAlpha () {
        return alpha;
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


    //Sum for all samples M
    //(-1/m) * SUM(  (Y * log(h)) + ((1-Y) * log(1-h))  )
    public double getCost() {
            //Sum for all samples M
            //(-1/m) * SUM(  (Y * log(h)) + ((1-Y) * log(1-h))  )

            //two vector cross products. gives two scalars. add them. mult by (-1/m)
                //Y * log(h)
                //(1-Y) * log(1-h)

        final int numSamples = getNumInstances();

        double cost = 0.0;

        for( int x=0;x<numSamples;x++) { //for every row( sample)

            //Assume an intercept independent var of 1.0 in the (virtual) first column of independent matrix.
            double hypothesis = thetas.getQuick(0); //Assumed intercept (1.0 * theta)
            for( int y=0;y<independent.columns();y++) { //for every indep. var (column)
                hypothesis += independent.getQuick(x,y) * thetas.getQuick(1+y); //+1 because of intercept
            }
            hypothesis = logit(hypothesis);

            double response  = dependent.getQuick(x);
            double subResult = (response * Math.log(hypothesis)) + ((1.0-response) * Math.log(1.0-hypothesis));
            cost += subResult;
        }

        cost *= (-1.0/numSamples);

//            if( isRegularized() ) {
//                double regularize = 0;
//                //(Lambda/(2m))* sum(theta^2)
//                //But sum is from 1 to N.. Skip intercept?
//                for( int x=0;x<thetas.size();x++) {
//                    regularize += ( thetas.getQuick(x) * thetas.getQuick(x) );
//                }
//                regularize *= (lambda/(2*getNumInstances()));
//                cost += regularize;
//            }

        return(cost);
    }

    public DoubleMatrix1D getGradient() {

        //TODO: once we get it working up to here.. we can drop the hypothesis matrix as well.. and getGradient in one step
        //hypothesis matrix is set to (h-y)
        //calcHypothesisError();
        //System.out.println("H:" + Arrays.toString(hypothesies.toArray()));

        if( isLogistic ) {
            //For Regularized LR, the first term (intercept) is not regularized.

            //delta is in hypothesis after calcHypothesisError()

            //delta = (1/m) * SUM( delta * xj )



            double[] tempDeltas = new double[deltas.size()];

            final int numSamples = getNumInstances();
            for( int x=0;x<numSamples;x++) { //for every row( sample)

                //Assume an intercept independent var of 1.0 in the (virtual) first column of independent matrix.
                double tempHypothesis = thetas.getQuick(0); //Assumed intercept (1.0 * theta)
                for( int y=0;y<independent.columns();y++) { //for every indep. var (column)
                    tempHypothesis += independent.getQuick(x,y) * thetas.getQuick(1+y); //+1 because of intercept
                }
                tempHypothesis = logit(tempHypothesis);

                //Now update hyp, with the delta between
                double response  = dependent.getQuick(x);
                tempHypothesis -= response;


                //hypothesies.setQuick(x,tempHypothesis);
                double error = tempHypothesis;
                tempDeltas[0] += error;
                for( int y=1;y<tempDeltas.length;y++) {
                    tempDeltas[y] += (error * independent.get(x,y-1) );
                }

            }

//
//            for( int x=0;x<getNumInstances();x++) {
//                double error = hypothesies.getQuick(x);
//                tempDeltas[0] += error;
//                for( int y=1;y<tempDeltas.length;y++) {
//                    tempDeltas[y] += (error * independent.get(x,y-1) );
//                }
//            }

            //scale by (1/m)
            for( int x=0;x<deltas.size();x++) {
                tempDeltas[x] *= 1.0 / (double)getNumInstances();
            }

            deltas.assign(tempDeltas);

            System.out.println("d:" + Arrays.toString(deltas.toArray()));
            //deltas.assign(Functions.mult(1.0/(double)getNumInstances()));

//            if( isRegularized() ) {
//
//                //For Regularized LR, the first term (intercept) is not regularized.
//                //Skip the intercept, and add regularization to all others.
//                double regularScale = lambda / (double)getNumInstances();
//                for( int x=1;x<deltas.size();x++) {
//                    //add (lambda/m)*thetaj
//                    double adjust = regularScale*thetas.getQuick(x);
//                    deltas.setQuick(x, deltas.getQuick(x) + adjust);
//                }
//            }

            System.out.println("D:" + Arrays.toString(deltas.toArray()));
            return( deltas );
        }

        return( null );
    }

    /**
     * Get the Current Theta parameters
     * @return
     */
    public DoubleMatrix1D getThetas () {
        return thetas;
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
    public int getNumInstances() {
        return( dependent.size() );
    }

    public int getNumParameters() {
        return( thetas.size() );
    }


    public boolean solve() {
        Logistic            opt       = new Logistic(this);
        LimitedMemoryBFGS   optimizer = new LimitedMemoryBFGS(opt);
        optimizer.setTolerance(.000001);




        boolean converged = false;

        try {
            converged = optimizer.optimize();
        } catch (Exception e) {
            System.out.println(e);
            // This exception may be thrown if L-BFGS
            //  cannot step in the current direction.
            // This condition does not necessarily mean that
            //  the optimizer has failed, but it doesn't want
            //  to claim to have succeeded...
        }
        return( converged );
    }

//
//    /**
//     * (Lambda/2M) SUM( Theta^2 )
//     * @return
//     */
//    public double getRegularizationValue () {
//        return( (lambda / (2.0*dependent.size())) * algebra.mult(thetas,thetas) );
//    }


    public static class Logistic implements Optimizable.ByGradientValue {
        private final GlmLess glm;

        private boolean  isValStale      = true;
        private double   cachedVal       = 0.0;

        private boolean  isGradientStale = true;
        private double[] cachedGradient  = null;

        public Logistic (GlmLess glm) {
            this.glm = glm;
        }

        @Override
        /**
         * I assume this is cost gradient for each parameter
         * get gradient and put it in to the input
         */
        public void getValueGradient (double[] outputGradient) {
            if (isGradientStale) {
                //Make copy
                double[] result = glm.getGradient().toArray();
                for (int x = 0; x < result.length; x++) {
                    result[x] = -result[x];
                }
                cachedGradient = result;
                isGradientStale = false;
            }
            System.arraycopy(cachedGradient, 0, outputGradient, 0, cachedGradient.length);
            System.out.println("g:"+ Arrays.toString(outputGradient));
        }

        @Override
        /**
         * I assume this is cost fn
         */
        public double getValue () {
            if( isValStale ) {
                isGradientStale = true;
                cachedVal = -glm.getCost();
                isValStale = false;
            }
            System.out.println("v:"+cachedVal);
            return (cachedVal);
        }

        @Override
        public int getNumParameters () {
            return (glm.getNumParameters());
        }

        @Override
        public void getParameters (double[] params) {
            DoubleMatrix1D thetas = glm.getThetas();
            for (int x = 0; x < thetas.size(); x++) {
                params[x] = thetas.getQuick(x);
            }
        }

        @Override
        public double getParameter (int i) {
            return (glm.getThetas().getQuick(i));
        }

        @Override
        public void setParameters (double[] params) {
            DoubleMatrix1D thetas = glm.getThetas();
            for (int x = 0; x < thetas.size(); x++) {
                thetas.setQuick(x, params[x]);
            }
            isValStale = true;
            isGradientStale = true;
        }

        @Override
        public void setParameter (int i, double val) {
            glm.getThetas().setQuick(i, val);
            isValStale = true;
            isGradientStale = true;
        }
    }
}
