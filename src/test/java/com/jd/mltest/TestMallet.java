package com.jd.mltest;

import cc.mallet.optimize.LimitedMemoryBFGS;
import cc.mallet.optimize.Optimizable;

import cc.mallet.optimize.tests.TestOptimizable;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import org.junit.Ignore;
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
    @Ignore
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

        System.out.println("Converged:" + converged);
        System.out.println("Cost:     " + glm.getCost());
        System.out.println("Gradient: " + glm.getGradient());
        System.out.println("Params:   " + Arrays.toString(glm.getThetas().toArray()));

        final double BIG_EPSILON =0.001;
        assertEquals(0.69024,glm.getCost(),EPSILON);

        assertEquals(-0.01418,glm.getThetas().toArray()[0],BIG_EPSILON);
        assertEquals(-0.3035,glm.getThetas().toArray()[1],BIG_EPSILON);
        assertEquals(-0.0168,glm.getThetas().toArray()[2],BIG_EPSILON);
    }


    @Test
    public void testMalletBig() {

        //39 seconds
//        That took 39418917
//        Converged:true
//        Cost:     7.211513133517897E-4
//        Gradient: 1 x 2 matrix
//        0.00072 0.000495
//        Params:   [-8.01682140693386, 2.4684312218964246]

        final double ALPHA          = .0001;
        final double LAMBDA         = 0.0;


        final int NUM_SAMPLES = 15000000;
        double[][] bigInd = new double[NUM_SAMPLES][2];
        double[] bigDep   = new double[NUM_SAMPLES];
        Random rand = new Random();
        for( int x=0;x<NUM_SAMPLES;x++) {
            boolean isResp = rand.nextBoolean();
            bigDep[x] = isResp?1:0;
            bigInd[x][0] = 1; //intercept
            bigInd[x][1] = isResp?(10+rand.nextDouble()):rand.nextDouble();
        }
        //rows,columns
        //Xi
        //These are the example data, i(down the column) is instance, j is each feature (across the row)
        DoubleMatrix2D independent      = new DenseDoubleMatrix2D(bigInd);

        //Yi
        //These are the results of the example linear equation.
        DoubleMatrix1D dependent        = new DenseDoubleMatrix1D(bigDep);
        //independent = mapFeature(independent);
        bigInd = null;
        bigDep = null;



        //Add all sort so interacitons
        //independent = mapFeature(independent);
        System.out.println("Go");
        long start = System.nanoTime();
        Glm glm = new Glm(independent,dependent,ALPHA,true, LAMBDA );

        //For testing start at all 0.
        glm.getThetas().assign(0);


        Logistic  opt       = new Logistic(glm);
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
        long end = System.nanoTime();
        System.out.println("That took " + (end-start)/1000 );

        System.out.println("Converged:" + converged);
        System.out.println("Cost:     " + glm.getCost());
        System.out.println("Gradient: " + glm.getGradient());
        System.out.println("Params:   " + Arrays.toString(glm.getThetas().toArray()));

//        final double BIG_EPSILON =0.001;
//        assertEquals(0.69024,glm.getCost(),EPSILON);
//
//        assertEquals(-0.01418,glm.getThetas().toArray()[0],BIG_EPSILON);
//        assertEquals(-0.3035,glm.getThetas().toArray()[1],BIG_EPSILON);
//        assertEquals(-0.0168,glm.getThetas().toArray()[2],BIG_EPSILON);
    }

    public static class Logistic implements Optimizable.ByGradientValue {
        private final Glm glm;

        private boolean  isValStale      = true;
        private double   cachedVal       = 0.0;

        private boolean  isGradientStale = true;
        private double[] cachedGradient  = null;

        public Logistic (Glm glm) {
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
