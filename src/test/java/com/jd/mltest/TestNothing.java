package com.jd.mltest;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.linalg.Algebra;
import cern.jet.math.Functions;
import org.junit.Test;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import static org.junit.Assert.*;

/**
 *
 */
public class TestNothing {

    @Test
    public void testNothing() {
        DoubleMatrix2D helloMatrix = new DenseDoubleMatrix2D(4,5);
        helloMatrix.assign(1D);
        //test
    }

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
    @Test
    public void testDescent() {
        final int    NUM_EXAMPLES = 8; //M
        final int    NUM_PARAMS   = 2; //N
        final double ALPHA        = .01;

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
            independent.set(x, 0, 0); //y=1
            independent.set(x, 1, 2 * x);
            independent.set(x, 2, 3 * x);
        }

        //initialize dependent Yi
        for( int x=0;x<NUM_EXAMPLES;x++) {
            dependent.set(x, (x + 1));
        }

        //Initialize Thetas to all 1.
        for( int x=0;x<NUM_PARAMS+1;x++) {
            thetas.set(x,1);
        }

        for( int x=0;x<20;x++) {
            thetas = descent( ALPHA, thetas, independent, dependent );
            System.out.println(thetas);
        }
    }


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

        //hypothesis - Y   (vector
        //Now we have for each Xi, the difference between predictect by the hypothesis and the actual Yi
        hypothesies.assign(dependent, Functions.minus );

        //Transpose Examples(MxN) to NxM so we can matrix multiply by hypothesis Nx1
        //Note that the Transpose is constant time and doesn't create a new matrix.
        DoubleMatrix2D transposed = algebra.transpose(independent);

        DoubleMatrix1D deltas     = algebra.mult(transposed, hypothesies );



        // Scale the deltas by 1/m and learning rate alhpa.  (alpha/m)
        deltas.assign(Functions.mult(modifier));

        //Theta = Theta - Deltas
        thetas.assign( deltas, Functions.minus );

        return( thetas );
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

        System.out.println(examples);

        //I think this can just skip the transpose of theta.
        DoubleMatrix1D result = algebra.mult( examples, thetas );
        System.out.println(result);
        //1+2+3
        //2+6+12
        //1+0+6
        assertEquals(result, new DenseDoubleMatrix1D(new double[]{6,
                                                                  20,
                                                                  7}));


    }

}
