package com.jd.mltest;

import cern.colt.function.DoubleFunction;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.linalg.Algebra;
import cern.jet.math.Functions;
import org.junit.Test;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;


import java.util.Random;

import static org.junit.Assert.assertEquals;

/**
 *
 */
public class TestSimpleDescent {
    private static final double EPSILON = 0.0001;


    /**
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
    @SuppressWarnings ("UnnecessaryLocalVariable")
    @Test
    public void testDescentMultiple() {
        final int    NUM_EXAMPLES   = 8; //M
        final int    NUM_PARAMS     = 2; //N
        final double ALPHA          = .01;
        final int    NUM_ITERATIONS = 100000;
        Random random = new Random();
        double w0 = 10.0;
        double w1 = .5;
        double w2 = (1.0/3.0);


        //These are the weights for linear regression (Theta or Beta depending on your preference)
        DoubleMatrix1D thetas           = new DenseDoubleMatrix1D(NUM_PARAMS+1);

        //rows,columns
        //Xi
        //These are the example data, i(down the column) is instance, j is each feature (across the row)
        DoubleMatrix2D independent      = new DenseDoubleMatrix2D(NUM_EXAMPLES,NUM_PARAMS+1);

        //Yi
        //These are the results of the example linear equation.
        DoubleMatrix1D dependent        = new DenseDoubleMatrix1D(NUM_EXAMPLES);

        //initialize Independent Xi
        //Going to create test data y= .5(x1) + .33(x2)
        for( int x=0;x<NUM_EXAMPLES;x++) {
            double x1 = random.nextDouble();
            double x2 = random.nextDouble();
            independent.set(x, 0, 1); //We always set this to 1 for the intercept
            independent.set(x, 1, x1);
            independent.set(x, 2, x2 );
        }

        //initialize dependent Yi
        for( int x=0;x<NUM_EXAMPLES;x++) {
            dependent.set(x, w0 +  (w1*independent.get(x,1)) + (w2*independent.get(x,2)) );
        }

//        System.out.println(independent);
//        System.out.println(dependent);

        //Initialize Thetas to all 1.
        for( int x=0;x<NUM_PARAMS+1;x++) {
            thetas.set(x,1);
        }

        for( int x=0;x<NUM_ITERATIONS;x++) {
            thetas = descent( ALPHA, thetas, independent, dependent );
//            if( x%1000 == 0) {
//                System.out.println(thetas);
//            }
        }

        //It seems like if we don't regularize to Zero mean, then the learning rate has to go way up or it goes off the
        //rails real quick.

        //TODO: Not sure why this isn't what I put in...
        assertEquals(w0,thetas.get(0), EPSILON);
        assertEquals(w1,thetas.get(1), EPSILON);
        assertEquals(w2,thetas.get(2), EPSILON);
    }


    @Test
    /**
     * test a simple y=.5x+7.0
     */
    public void testLinearDescentIntercept() {
        final int    NUM_EXAMPLES   = 8; //M
        final int    NUM_PARAMS     = 1; //N
        final double ALPHA          = .01;
        final int    NUM_ITERATIONS = 10000;

        //These are the weights for linear regression (Theta or Beta depending on your preference)
        DoubleMatrix1D thetas           = new DenseDoubleMatrix1D(NUM_PARAMS+1);

        //rows,columns
        //Xi
        //These are the example data, i(down the column) is instance, j is each feature (across the row)
        DoubleMatrix2D independent      = new DenseDoubleMatrix2D(NUM_EXAMPLES,NUM_PARAMS+1);

        //Yi
        //These are the results of the example linear equation.
        DoubleMatrix1D dependent        = new DenseDoubleMatrix1D(NUM_EXAMPLES);

        //initialize Independent Xi
        //Going to create test data y= .5(x1) + 7.0
        for( int x=0;x<NUM_EXAMPLES;x++) {
            independent.set(x, 0, 1); //We always set this to 1 for the intercept
            independent.set(x, 1, (double)x);
        }

        //initialize dependent Yi
        for( int x=0;x<NUM_EXAMPLES;x++) {
            dependent.set(x, 7.0+(.5D*(double)x) );
        }

        //Initialize Thetas to all 1.
        for( int x=0;x<NUM_PARAMS+1;x++) {
            thetas.set(x,1);
        }

        for( int x=0;x<NUM_ITERATIONS;x++) {
            thetas = descent( ALPHA, thetas, independent, dependent );
        }

        //0 intercept
        assertEquals(7.0,thetas.get(0), EPSILON);
        //.5x
        assertEquals(0.5,thetas.get(1), EPSILON);
    }

    /**
     * sum((X * theta - y) .* X(:, i)) ./ m;
     * @param alpha Learning Rate
     * @param thetas Current Thetas
     * @param independent
     * @param dependent
     * @return new Thetas
     */
    public DoubleMatrix1D descent(double         alpha,
                                  DoubleMatrix1D thetas,
                                  DoubleMatrix2D independent,
                                  DoubleMatrix1D dependent ) {
        Algebra algebra     = new Algebra();

        // ALPHA*(1/M) in one.
        double  modifier    = alpha / (double)independent.rows();

        //I think this can just skip the transpose of theta.
        //This is the result of every Xi run through the theta (hypothesis fn)
        //So each Xj feature is multiplied by its Theata, to get the results of the hypotesis
        DoubleMatrix1D hypothesies = algebra.mult( independent, thetas );

        //hypothesis - Y
        //Now we have for each Xi, the difference between predicted by the hypothesis and the actual Yi
        hypothesies.assign(dependent, Functions.minus);


        //Transpose Examples(MxN) to NxM so we can matrix multiply by hypothesis Nx1
        //Note that the Transpose is constant time and doesn't create a new matrix.
        DoubleMatrix2D transposed = algebra.transpose(independent);

        DoubleMatrix1D deltas     = algebra.mult(transposed, hypothesies );



        // Scale the deltas by 1/m and learning rate alhpa.  (alpha/m)
        //deltas.assign(Functions.mult(modifier));

        //Theta = Theta - Deltas
        //thetas.assign( deltas, Functions.minus );

        // thetas = thetas - (deltas*modifier)  in one step
        thetas.assign(deltas, Functions.minusMult(modifier));


        return( thetas );
    }


    /**
     * sum((X * theta - y) .* X(:, i)) ./ m;
     * @param alpha Learning Rate
     * @param thetas Current Thetas
     * @param independent
     * @param dependent
     * @return new Thetas
     */
    public DoubleMatrix1D logisticDescent(double         alpha,
                                          DoubleMatrix1D thetas,
                                          DoubleMatrix2D independent,
                                          DoubleMatrix1D dependent ) {
        Algebra algebra     = new Algebra();

        // ALPHA*(1/M) in one.
        //double  modifier    = alpha / (double)independent.rows();

        //hypothesis is 1/( 1+ e ^ (theta(Transposed) * X))
        DoubleMatrix1D hypothesies = algebra.mult( independent, thetas );

        //h = 1/(1+ e^h)
        hypothesies.assign(new DoubleFunction() {
            @Override
            public double apply (double val) {
                return( logit( val ) );
            }
        });

        //hypothesis - Y
        //Now we have for each Xi, the difference between predicted by the hypothesis and the actual Yi
        hypothesies.assign(dependent, Functions.minus);


        //Transpose Examples(MxN) to NxM so we can matrix multiply by hypothesis Nx1
        //Note that the Transpose is constant time and doesn't create a new matrix.
        DoubleMatrix2D transposed = algebra.transpose(independent);

        DoubleMatrix1D deltas     = algebra.mult(transposed, hypothesies );


        // thetas = thetas - (deltas*alpha)  in one step
        thetas.assign(deltas, Functions.minusMult(alpha));


        return( thetas );
    }

    @Test
    public void testLogisticDescentMultiple() {
        //Cost function: -y * log(h(x)) - (1-y)log(1-h(x))
        //(-1/m) Sum(Cost)
        final int    NUM_EXAMPLES   = 1000; //M
        final int    NUM_PARAMS     = 2; //N
        final double ALPHA          = .001;
        final int    NUM_ITERATIONS = 100000;
        //final int    PRINT_AT        = 10000;
        Random random = new Random();
        double w0 = 6.0;
        double w1 = .5;
        double w2 = (1.0/3.0);


        //These are the weights for linear regression (Theta or Beta depending on your preference)
        DoubleMatrix1D thetas           = new DenseDoubleMatrix1D(NUM_PARAMS+1);

        //rows,columns
        //Xi
        //These are the example data, i(down the column) is instance, j is each feature (across the row)
        DoubleMatrix2D independent      = new DenseDoubleMatrix2D(NUM_EXAMPLES,NUM_PARAMS+1);

        //Yi
        //These are the results of the example linear equation.
        DoubleMatrix1D dependent        = new DenseDoubleMatrix1D(NUM_EXAMPLES);

        //initialize Independent Xi
        //Going to create test data y= .5(x1) + .33(x2) + 10
        for( int x=0;x<NUM_EXAMPLES;x++) {
            double x1 = random.nextDouble();
            double x2 = random.nextDouble();

            if( random.nextBoolean() ) {
                x1 *= -1.0;
            }
            if( random.nextBoolean() ) {
                x2 *= -1.0;
            }

            independent.set(x, 0, 1); //We always set this to 1 for the intercept
            independent.set(x, 1, x1);
            independent.set(x, 2, x2 );
        }

        //initialize dependent Yi
        for( int x=0;x<NUM_EXAMPLES;x++) {
            double val      = w0 +  (w1*independent.get(x,1)) + (w2*independent.get(x,2));
            double logitVal = logit( val );
//            if( logitVal < 0.5 ) {
//                logitVal = 0;
//            }else {
//                logitVal = 1;
//            }
            dependent.set(x, logitVal );
        }

//        System.out.println(independent);
//        System.out.println(dependent);

        //Initialize Thetas to all 1.
        for( int x=0;x<NUM_PARAMS+1;x++) {
            thetas.set(x,1);
        }

        for( int x=0;x<NUM_ITERATIONS;x++) {
            thetas = logisticDescent( ALPHA, thetas, independent, dependent );
//            if( x%PRINT_AT == 0) {
//                System.out.println(thetas);
//            }
        }

        //It seems like if we don't regularize to Zero mean, then the learning rate has to go way up or it goes off the
        //rails real quick.

        //TODO: Not sure why this isn't what I put in...
        assertEquals(w0,thetas.get(0), EPSILON);
        assertEquals(w1,thetas.get(1), EPSILON);
        assertEquals(w2,thetas.get(2), EPSILON);
    }

    @Test
    public void testLogistic() {


    }

    public static double logit( double val ) {
        return( 1.0 / (1.0 + Math.exp(-val)));
    }


    @Test
    /**
     * Make sure colt is working like I think it is.
     */
    public void testMult() {
        Algebra algebra = new Algebra();

        //These are the weights for linear regression
        DoubleMatrix1D thetas           = new DenseDoubleMatrix1D(new double[]{1,
                                                                               2,
                                                                               3});


        DoubleMatrix2D examples         = new DenseDoubleMatrix2D(new double[][]{{1,1,1},
                                                                                 {2,3,4},
                                                                                 {1,0,2}});

        //System.out.println(examples);

        //I think this can just skip the transpose of theta.
        DoubleMatrix1D result = algebra.mult( examples, thetas );
        //System.out.println(result);
        //1+2+3
        //2+6+12
        //1+0+6
        assertEquals(result, new DenseDoubleMatrix1D(new double[]{6,
                                                                  20,
                                                                  7}));


    }

}



//10,0,0
//11.166667,1,2
//12.333333,2,4
//13.5,3,6
//14.666667,4,8
//15.833333,5,10
//17,6,12
//18.166667,7,14
