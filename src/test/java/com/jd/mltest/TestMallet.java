package com.jd.mltest;

import cc.mallet.optimize.LimitedMemoryBFGS;
import cc.mallet.optimize.Optimizable;

import cc.mallet.optimize.tests.TestOptimizable;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import org.junit.Test;


import java.util.Arrays;
import java.util.Random;

import static com.jd.mltest.TestSimpleDescent.*;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Jumping ahead to using someone else's LBFGS routine...
 * I'm going to circle back to implementing Linear/logistic and finally NeuralNets
 * using an optimization package but specifying cost and gradient functions.
 */
public class TestMallet {

    @Test
    public void testMallet() {
        final double ALPHA          = .001;
        final Double LAMBDA         = 0D;


        //rows,columns
        //Xi
        //These are the example data, i(down the column) is instance, j is each feature (across the row)
        DoubleMatrix2D independent      = new DenseDoubleMatrix2D(getIndep(getFirstTestData(),true));

        //Yi
        //These are the results of the example linear equation.
        DoubleMatrix1D dependent        = new DenseDoubleMatrix1D(getDep(getFirstTestData()));

        Glm glm = new Glm(independent,dependent,ALPHA,true, LAMBDA );

        //For testing start at all 0.
        glm.getThetas().assign(0);


        Logistic            opt       = new Logistic(glm);
        LimitedMemoryBFGS   optimizer = new LimitedMemoryBFGS(opt);

        //TestOptimizable.testGetSetParameters( opt );

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

        System.out.println("Converged:" + converged);
        System.out.println("Cost:     " + glm.getCost());
        System.out.println("Gradient: " + glm.getGradient());
        System.out.println("Params:   " + glm.getThetas());

        assertEquals(-25.16133356066622,glm.getThetas().get(0), EPSILON);
        assertEquals(0.20623171324620174,glm.getThetas().get(1), EPSILON);
        assertEquals(0.20147160039363188, glm.getThetas().get(2), EPSILON);

        assertEquals(.7762906907271161,glm.predict(45,85), EPSILON);

        //System.out.println(opt.getParameter(0) + ", " + opt.getParameter(1)  );
    }

    @Test
    public void testMalletUnit() {
        final double ALPHA          = .001;
        final Double LAMBDA         = 0D;


        //rows,columns
        //Xi
        //These are the example data, i(down the column) is instance, j is each feature (across the row)
        DoubleMatrix2D independent      = new DenseDoubleMatrix2D(getIndep(getFirstTestData(),true));

        //Yi
        //These are the results of the example linear equation.
        DoubleMatrix1D dependent        = new DenseDoubleMatrix1D(getDep(getFirstTestData()));

        Glm glm = new Glm(independent,dependent,ALPHA,true, LAMBDA );

        //For testing start at all 0.
        glm.getThetas().assign(0);


        Logistic            opt       = new Logistic(glm);
        //LimitedMemoryBFGS   optimizer = new LimitedMemoryBFGS(opt);

        assertTrue(TestOptimizable.testGetSetParameters(opt));
        assertTrue(TestOptimizable.testValueAndGradient(opt));
        assertTrue(TestOptimizable.testValueAndGradientRandomParameters(opt, new Random()));


    }

    @Test
    public void testMalletTwo() {
        final double ALPHA          = .001;
        final double LAMBDA         = 0.0;


        //rows,columns
        //Xi
        //These are the example data, i(down the column) is instance, j is each feature (across the row)
        DoubleMatrix2D independent      = new DenseDoubleMatrix2D(getIndep(getData2(),true));

        //Yi
        //These are the results of the example linear equation.
        DoubleMatrix1D dependent        = new DenseDoubleMatrix1D(getDep(getData2()));



        //Add all sort so interacitons
        //independent = mapFeature(independent);

        Glm glm = new Glm(independent,dependent,ALPHA,true, LAMBDA );

        //For testing start at all 0.
        glm.getThetas().assign(0);


        Logistic  opt       = new Logistic(glm);
        LimitedMemoryBFGS   optimizer = new LimitedMemoryBFGS(opt);
        optimizer.setTolerance(.000000000000001);




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

        System.out.println("Converged:" + converged);
        System.out.println("Cost:     " + glm.getCost());
        System.out.println("Gradient: " + glm.getGradient());
        System.out.println("Params:   " + Arrays.toString(glm.getThetas().toArray()));


//Cost:     0.609940054943988
//Gradient: 1 x 28 matrix
//0.000577 0.001266 0.00033 0.001266 0.00033 0.000555 0.001266 0.00033 0.000555 0.000347 0.001266 0.00033 0.000555 0.000347 0.000466 0.001266 0.00033 0.000555 0.000347 0.000466 0.000367 0.001266 0.00033 0.000555 0.000347 0.000466 0.000367 0.000375
//Params:   1 x 28 matrix
//0.214821 0.081307 0.017729 0.081307 0.017729 -0.481613 0.081307 0.017729 -0.481613 0.170024 0.081307 0.017729 -0.481613 0.170024 -0.565344 0.081307 0.017729 -0.481613 0.170024 -0.565344 -0.078423 0.081307 0.017729 -0.481613 0.170024 -0.565344 -0.078423 -0.492659

//No Reg.
//        Cost:     0.5756399307111906
//        Gradient: 1 x 28 matrix
//        0.000121 0.000121 0.000155 0.000121 0.000155 4.230738E-005 0.000121 0.000155 4.230738E-005 -3.664739E-005 0.000121 0.000155 4.230738E-005 -3.664739E-005 -0.000217 0.000121 0.000155 4.230738E-005 -3.664739E-005 -0.000217 0.000278 0.000121 0.000155 4.230738E-005 -3.664739E-005 -0.000217 0.000278 6.66527E-005
//        Params:   1 x 28 matrix
//        0.058781 0.058781 -0.292555 0.058781 -0.292555 0.581333 0.058781 -0.292555 0.581333 2.095154 0.058781 -0.292555 0.581333 2.095154 -3.997081 0.058781 -0.292555 0.581333 2.095154 -3.997081 -1.232733 0.058781 -0.292555 0.581333 2.095154 -3.997081 -1.232733 -4.8656

//        Cost:     0.6902412268833071
//        Gradient: 1 x 3 matrix
//        0.000117 -4.615523E-005 0.000114
//        Params:   1 x 3 matrix
//        -0.013915 -0.304199 -0.016838
        //System.out.println(opt.getParameter(0) + ", " + opt.getParameter(1)  );
    }

    public static class Logistic implements Optimizable.ByGradientValue {
        private final Glm glm;

        public Logistic (Glm glm) {
            this.glm = glm;
        }

        @Override
        /**
         * I assume this is cost gradient for each parameter
         * get gradient and put it in to the input
         */
        public void getValueGradient (double[] gradient) {
            double[] result = glm.getGradient().toArray();
            System.arraycopy(result, 0, gradient, 0, result.length );
            for( int x=0;x<result.length;x++) {
                gradient[x] = -gradient[x];
            }
        }

        @Override
        /**
         * I assume this is cost fn
         */
        public double getValue () {
            //return glm.getCost();
            return -glm.getCost();
        }

        @Override
        public int getNumParameters () {
            return( glm.getNumParameters() );
        }

        @Override
        public void getParameters (double[] params) {
            DoubleMatrix1D thetas = glm.getThetas();
            for( int x=0;x<thetas.size();x++) {
                params[x] = thetas.getQuick(x);
            }
        }

        @Override
        public double getParameter (int i) {
            return( glm.getThetas().getQuick(i) );
        }

        @Override
        public void setParameters (double[] params) {
            DoubleMatrix1D thetas = glm.getThetas();
            for( int x=0;x<thetas.size();x++) {
                thetas.setQuick(x,params[x]);
            }
        }

        @Override
        public void setParameter (int i, double val) {
            glm.getThetas().setQuick(i,val);
        }
    }
}
