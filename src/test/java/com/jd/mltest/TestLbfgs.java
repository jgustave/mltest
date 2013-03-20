package com.jd.mltest;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import lbfgsb.*;
import org.junit.Test;

import static com.jd.mltest.TestSimpleDescent.getDep;
import static com.jd.mltest.TestSimpleDescent.getFirstTestData;
import static com.jd.mltest.TestSimpleDescent.getIndep;
import static org.junit.Assert.assertEquals;

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
    public void testLbfgs() {
        //ADD -Djava.library.path=/Users/jerdavis/Dropbox/devhome/mltest/lib
        SampleRun.main(new String[]{});
    }

    @Test
    public void testFoo() {
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

            assertEquals(-25.16133356066622,glm.getThetas().get(0), EPSILON);
            assertEquals(0.20623171324620174,glm.getThetas().get(1), EPSILON);
            assertEquals(0.20147160039363188, glm.getThetas().get(2), EPSILON);

            assertEquals(.7762906907271161,glm.predict(45,85), EPSILON);


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
                glm.getThetas().set(x,point[x]);
            }
            double   cost     = glm.getCost();
            double[] gradient = glm.getGradient().toArray();

//            cost = -cost;
//            for( int x=0;x<gradient.length;x++) {
//                gradient[x] = -gradient[x];
//            }
            FunctionValues vals =  new FunctionValues(cost, gradient);
            return( vals );
        }
    }
}
