package com.jd.mltest;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import lbfgsb.*;
import org.junit.Test;

import java.util.Arrays;
import java.util.Random;

import static com.jd.mltest.TestSimpleDescent.*;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 *https://github.com/mkobos/lbfgsb_wrapper

 install gfortran
 http://hpc.sourceforge.net
 http://sourceforge.net/projects/hpc/files/hpc/g95/gfortran-mlion.tar.gz
 copy all the stuff in to /usr/local

 sudo port install swig
 sudo port install swig-java

 jni.h can't be found.. den if you set JAVA_HOME
 export JAVA_HOME=/Library/Java/Home


 jni.h lives in:
 /System/Library/Frameworks/JavaVM.framework/Headers

 ln -s /System/Library/Frameworks/JavaVM.framework/Headers /Library/Java/Home/include

 go to the /dist dir
 cp liblbfgsb_wrapper.so liblbfgsb_wrapper.dylib

 */
public class TestLbfgs {
    public static final double EPSILON = 0.0001;

    @Test
    /**
     * -Djava.library.path=/Users/jerdavis/Dropbox/devhome/mltest/lib
     */
    public void testOpt1() {
        try
        {
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

            Minimizer alg = new Minimizer();
            alg.setNoBounds(glm.getNumParameters());


            GlmFunc glmFunc = new GlmFunc(glm);
            Result  result  = alg.run(glmFunc, glm.getThetas().toArray() );

            System.out.println("The final result: "+result);
            System.out.println("Cost:     " + glm.getCost());
            System.out.println("Gradient: " + glm.getGradient());
            System.out.println("Params:   " + glm.getThetas());
            assertTrue(IterationsInfo.StopType.ABNORMAL != result.iterationsInfo.type);


            assertEquals(-25.16133356066622,glm.getThetas().get(0), EPSILON);
            assertEquals(0.20623171324620174,glm.getThetas().get(1), EPSILON);
            assertEquals(0.20147160039363188, glm.getThetas().get(2), EPSILON);

            assertEquals(.7762906907271161,glm.predict(45,85), EPSILON);

//            Cost:     0.20349770158981684
//            Gradient: 1 x 3 matrix
//            -1.543441E-007 -7.159654E-006 -1.183759E-005
//            Params:   1 x 3 matrix
//            -25.161338 0.206232 0.201472
        } catch (LBFGSBException e) {
            e.printStackTrace();
        }
    }

    @Test
    public void testOptTwo() {
        try
        {
            final double ALPHA          = .0001;
            final Double LAMBDA         = 0D;


            //rows,columns
            //Xi
            //These are the example data, i(down the column) is instance, j is each feature (across the row)
            DoubleMatrix2D independent      = new DenseDoubleMatrix2D(getIndep(getData2(),true));

            //Yi
            //These are the results of the example linear equation.
            DoubleMatrix1D dependent        = new DenseDoubleMatrix1D(getDep(getData2()));
            //independent = mapFeature(independent);

            Glm glm = new Glm(independent,dependent,ALPHA,true, LAMBDA );

            //For testing start at all 0.
            glm.getThetas().assign(0);

            Minimizer alg = new Minimizer();
            alg.setNoBounds(glm.getNumParameters());
            alg.setCorrectionsNo(20);


            GlmFunc glmFunc = new GlmFunc(glm);
            Result  result  = alg.run(glmFunc, glm.getThetas().toArray() );

            System.out.println("The final result: "+result);
            System.out.println("Cost:     " + glm.getCost());
            System.out.println("Gradient: " + glm.getGradient());
            System.out.println("Params:   " + Arrays.toString(glm.getThetas().toArray()));
            assertTrue(IterationsInfo.StopType.ABNORMAL != result.iterationsInfo.type);

//Cost:     0.609940054943988
//Gradient: 1 x 28 matrix
//0.000577 0.001266 0.00033 0.001266 0.00033 0.000555 0.001266 0.00033 0.000555 0.000347 0.001266 0.00033 0.000555 0.000347 0.000466 0.001266 0.00033 0.000555 0.000347 0.000466 0.000367 0.001266 0.00033 0.000555 0.000347 0.000466 0.000367 0.000375
//Params:   1 x 28 matrix
//0.214821 0.081307 0.017729 0.081307 0.017729 -0.481613 0.081307 0.017729 -0.481613 0.170024 0.081307 0.017729 -0.481613 0.170024 -0.565344 0.081307 0.017729 -0.481613 0.170024 -0.565344 -0.078423 0.081307 0.017729 -0.481613 0.170024 -0.565344 -0.078423 -0.492659

            //No Reg
//            Cost:     0.5713954703458933
//            Gradient: 1 x 28 matrix
//            0.000412 0.000412 -0.000168 0.000412 -0.000168 -5.115719E-005 0.000412 -0.000168 -5.115719E-005 0.000174 0.000412 -0.000168 -5.115719E-005 0.000174 0.000372 0.000412 -0.000168 -5.115719E-005 0.000174 0.000372 0.000198 0.000412 -0.000168 -5.115719E-005 0.000174 0.000372 0.000198 0.000407
//            Params:   1 x 28 matrix
//            0.095941 0.095941 -0.437031 0.095941 -0.437031 -0.705585 0.095941 -0.437031 -0.705585 3.671789 0.095941 -0.437031 -0.705585 3.671789 3.423657 0.095941 -0.437031 -0.705585 3.671789 3.423657 -6.386469 0.095941 -0.437031 -0.705585 3.671789 3.423657 -6.386469 -21.766948


            final double BIG_EPSILON =0.01;
            assertEquals(0.69024,glm.getCost(),EPSILON);

            assertEquals(-0.01418,glm.getThetas().toArray()[0],BIG_EPSILON);
            assertEquals(-0.3035,glm.getThetas().toArray()[1],BIG_EPSILON);
            assertEquals(-0.0168,glm.getThetas().toArray()[2],BIG_EPSILON);


        } catch (LBFGSBException e) {
            e.printStackTrace();
        }
    }


    @Test
    /**
     * Test 15M univariate.
     */
    public void testOptBig() {
        try
        {
            //50 sec
//            The final result: point=[-12.977191021560426,2.4849316593470094], functionValue=6.406705355890922E-6, gradient=[3.833090177458026E-6,-9.743645223038823E-6], iterationsInfo=(iterations=19, functionEvaluations=22, stopType=OTHER_STOP_CONDITIONS, stateDescription=CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL)
//            Cost:     6.406705355890922E-6
//            Gradient: 1 x 2 matrix
//            3.83309E-006 -9.743645E-006
//            Params:   [-12.977191021560426, 2.4849316593470094]
            final double ALPHA          = .0001;
            final Double LAMBDA         = 0D;

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
            System.out.println("Start opt");
            long start = System.nanoTime();
            Glm glm = new Glm(independent,dependent,ALPHA,true, LAMBDA );

            //For testing start at all 0.
            glm.getThetas().assign(0);

            Minimizer alg = new Minimizer();
            alg.setNoBounds(glm.getNumParameters());
            alg.setCorrectionsNo(20);


            GlmFunc glmFunc = new GlmFunc(glm);
            Result  result  = alg.run(glmFunc, glm.getThetas().toArray() );

            long end = System.nanoTime();
            System.out.println("That took"+ (end-start)/1000 );

            System.out.println("The final result: "+result);
            System.out.println("Cost:     " + glm.getCost());
            System.out.println("Gradient: " + glm.getGradient());
            System.out.println("Params:   " + Arrays.toString(glm.getThetas().toArray()));
            assertTrue(IterationsInfo.StopType.ABNORMAL != result.iterationsInfo.type);

//Cost:     0.609940054943988
//Gradient: 1 x 28 matrix
//0.000577 0.001266 0.00033 0.001266 0.00033 0.000555 0.001266 0.00033 0.000555 0.000347 0.001266 0.00033 0.000555 0.000347 0.000466 0.001266 0.00033 0.000555 0.000347 0.000466 0.000367 0.001266 0.00033 0.000555 0.000347 0.000466 0.000367 0.000375
//Params:   1 x 28 matrix
//0.214821 0.081307 0.017729 0.081307 0.017729 -0.481613 0.081307 0.017729 -0.481613 0.170024 0.081307 0.017729 -0.481613 0.170024 -0.565344 0.081307 0.017729 -0.481613 0.170024 -0.565344 -0.078423 0.081307 0.017729 -0.481613 0.170024 -0.565344 -0.078423 -0.492659

            //No Reg
//            Cost:     0.5713954703458933
//            Gradient: 1 x 28 matrix
//            0.000412 0.000412 -0.000168 0.000412 -0.000168 -5.115719E-005 0.000412 -0.000168 -5.115719E-005 0.000174 0.000412 -0.000168 -5.115719E-005 0.000174 0.000372 0.000412 -0.000168 -5.115719E-005 0.000174 0.000372 0.000198 0.000412 -0.000168 -5.115719E-005 0.000174 0.000372 0.000198 0.000407
//            Params:   1 x 28 matrix
//            0.095941 0.095941 -0.437031 0.095941 -0.437031 -0.705585 0.095941 -0.437031 -0.705585 3.671789 0.095941 -0.437031 -0.705585 3.671789 3.423657 0.095941 -0.437031 -0.705585 3.671789 3.423657 -6.386469 0.095941 -0.437031 -0.705585 3.671789 3.423657 -6.386469 -21.766948




        } catch (LBFGSBException e) {
            e.printStackTrace();
        }
    }

    //TODO: I think we want the point the be the Theta vector, and
    //we will cal the cost and gradient and put in FunctionValues

    public static class GlmFunc
            implements DifferentiableFunction{
        private final Glm glm;

        public GlmFunc (Glm glm) {
            this.glm = glm;
        }

        public FunctionValues getValues(double[] point){

            for( int x=0;x<point.length;x++) {
                glm.getThetas().setQuick(x, point[x]);
            }
            double   cost     = glm.getCost();
            double[] gradient = glm.getGradient().toArray();

            FunctionValues vals =  new FunctionValues(cost, gradient);
            return( vals );
        }
    }
}
