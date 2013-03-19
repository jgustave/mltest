package com.jd.mltest;

import cc.mallet.optimize.LimitedMemoryBFGS;
import cc.mallet.optimize.Optimizable;
import cc.mallet.optimize.Optimizer;
import cc.mallet.util.MalletLogger;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import org.junit.Test;

import java.util.logging.Level;
import java.util.logging.Logger;

import static com.jd.mltest.TestSimpleDescent.*;
import static org.junit.Assert.assertEquals;

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
    public void testMalletTwo() {
        final double ALPHA          = .001;
        final double LAMBDA         = 1.0;


        //rows,columns
        //Xi
        //These are the example data, i(down the column) is instance, j is each feature (across the row)
        DoubleMatrix2D independent      = new DenseDoubleMatrix2D(getIndep(getData2(),true));

        //Yi
        //These are the results of the example linear equation.
        DoubleMatrix1D dependent        = new DenseDoubleMatrix1D(getDep(getData2()));



        //Add all sort so interacitons
        independent = mapFeature(independent);

        Glm glm = new Glm(independent,dependent,ALPHA,true, LAMBDA );

        //For testing start at all 0.
        glm.getThetas().assign(0);


        Logistic  opt       = new Logistic(glm);
        LimitedMemoryBFGS   optimizer = new LimitedMemoryBFGS(opt);
        optimizer.setTolerance(.00000000001);


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

        //System.out.println(opt.getParameter(0) + ", " + opt.getParameter(1)  );
    }


    @Test
    public void testMalletMicro() {
        final double ALPHA          = .001;

        Logger log = Logger.getLogger(LimitedMemoryBFGS.class.getName());
        log.setLevel(Level.FINE);

        MalletLogger.getLogger("").setLevel(Level.FINE);
        MalletLogger.getLogger(LimitedMemoryBFGS.class.getName()).setLevel(Level.FINE);
        MalletLogger.getLogger(LimitedMemoryBFGS.class.getName()).config(
                "handlers = java.util.logging.ConsoleHandler\n" +
                ".level = ALL\n" +
                "\n" +
                "\n" +
                "# Console Logging\n" +
                "java.util.logging.ConsoleHandler.level = ALL\n");
        //rows,columns
        //Xi
        //These are the example data, i(down the column) is instance, j is each feature (across the row)

        DoubleMatrix2D independent      = new DenseDoubleMatrix2D(new double[][]{{1.0}});

        //Yi
        //These are the results of the example linear equation.
        DoubleMatrix1D dependent        = new DenseDoubleMatrix1D(new double[]{1.0});


        Glm glm = new Glm(independent,dependent,ALPHA,true, 1.0 );

        //For testing start at all 0.
        glm.getThetas().assign(0);


        Logistic  opt       = new Logistic(glm);
        Optimizer optimizer = new LimitedMemoryBFGS(opt);


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


        //System.out.println(opt.getParameter(0) + ", " + opt.getParameter(1)  );
    }


    public static class Logistic implements Optimizable.ByGradientValue {

//        private final DoubleMatrix2D independent;
//        private final DoubleMatrix1D dependent;
//        private final DoubleMatrix1D thetas;
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
            //For Regularized LR, the first term (intercept) is not regularized.


            //(1/m) * SUM( delta * xj)

            //For all others add regularize : (Lambda/m)*Thetaj
            double[] result = glm.getGradient().toArray();
            if( result.length != gradient.length ) {
                throw new RuntimeException("size mismatch");
            }
            System.arraycopy(result, 0, gradient, 0, result.length );
//            for( int x=0;x<gradient.length;x++) {
//                gradient[x] = -gradient[x];
//            }
        }

        @Override
        /**
         * I assume this is cost fn
         */
        public double getValue () {
            return glm.getCost();
            //return -glm.getCost();
        }

        @Override
        public int getNumParameters () {
            return( glm.getNumParameters() );
        }

        @Override
        public void getParameters (double[] params) {
            DoubleMatrix1D thetas = glm.getThetas();
            final int size = thetas.size();

            for( int x=0;x<size;x++) {
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
            final int size = thetas.size();

            for( int x=0;x<size;x++) {
                thetas.setQuick(x,params[x]);
            }
        }

        @Override
        public void setParameter (int i, double val) {
            glm.getThetas().setQuick(i,val);
        }
    }
}
