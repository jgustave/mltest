package com.jd.mltest;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.linalg.Algebra;

import cern.colt.matrix.linalg.SeqBlas;
import org.junit.Test;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;


import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertEquals;

/**
 * Just the proof of concept area.. No classes yet.
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
    public void testLinearDescentMultiple() {
        final int    NUM_EXAMPLES   = 8; //M
        final int    NUM_PARAMS     = 2; //N
        final double ALPHA          = .01;
        final int    NUM_ITERATIONS = 100000;
        Random random = new Random();
        double w0 = 10.0;
        double w1 = .5;
        double w2 = (1.0/3.0);

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
            independent.set(x, 0, 1); //We always set this to 1 for the intercept
            independent.set(x, 1, random.nextGaussian());
            independent.set(x, 2, random.nextGaussian() );
        }

        //initialize dependent Yi
        for( int x=0;x<NUM_EXAMPLES;x++) {
            dependent.set(x, w0 +  (w1*independent.get(x,1)) + (w2*independent.get(x,2)) );
        }

        Glm glm = new Glm(independent,dependent,ALPHA, false );
        //glm.scaleInputs();


        for( int x=0;x<NUM_ITERATIONS;x++) {
            glm.step();
            //thetas = linearDescent(ALPHA, thetas, independent, dependent);
            if( x%1000 == 0) {
                System.out.println(glm.getThetas());
                System.out.println(glm.getCost());
            }
        }

        //It seems like if we don't regularize to Zero mean, then the learning rate has to go way up or it goes off the
        //rails real quick.

        DoubleMatrix1D thetas = glm.getThetas();

        assertEquals(w0,thetas.get(0), EPSILON);
        assertEquals(w1,thetas.get(1), EPSILON);
        assertEquals(w2,thetas.get(2), EPSILON);
    }



    public static double logit( double val ) {
        return( 1.0 / (1.0 + Math.exp(-val)));
    }

    public static double evalLogistic( double[] parameters, double[] dependent ) {
        double sum = 0;
        for( int x=0;x<parameters.length;x++) {
            sum += parameters[x] * dependent[x];
        }
        return( logit( sum ) );
    }

    @Test
    public void testLogisticDescentMultiple() {
        //Cost function: -y * log(h(x)) - (1-y)log(1-h(x))
        //(-1/m) Sum(Cost)
        final int    NUM_EXAMPLES   = 100; //M
        final int    NUM_PARAMS     = 2; //N
        final double ALPHA          = .00001;
        final int    NUM_ITERATIONS = 100000000;
        final int    PRINT_AT       = 1000000;

        //rows,columns
        //Xi
        //These are the example data, i(down the column) is instance, j is each feature (across the row)
        DoubleMatrix2D independent      = new DenseDoubleMatrix2D(NUM_EXAMPLES,NUM_PARAMS+1);

        //Yi
        //These are the results of the example linear equation.
        DoubleMatrix1D dependent        = new DenseDoubleMatrix1D(NUM_EXAMPLES);

        initLogisticTest( independent, dependent );

        Glm glm = new Glm(independent,dependent,ALPHA,true);


        for( int x=0;x<NUM_ITERATIONS;x++) {
            //thetas = logisticDescent( ALPHA, thetas, independent, dependent );
            glm.step();
            if( x%PRINT_AT == 0) {
                System.out.println(glm.getThetas());
                System.out.println(glm.getCost());
            }
        }

        //It seems like if we don't regularize to Zero mean, then the learning rate has to go way up or it goes off the
        //rails real quick.


        assertEquals(-25.16133356066622,glm.getThetas().get(0), EPSILON);
        assertEquals(0.20623171324620174,glm.getThetas().get(1), EPSILON);
        assertEquals(0.20147160039363188, glm.getThetas().get(2), EPSILON);

        assertEquals(.7762906907271161,glm.predict(45,85), EPSILON);
    }


    @Test
    /**
     * Now we have feature scaled and get the same results.
     */
    public void testLogisticDescentMultipleScaled() {
        //Cost function: -y * log(h(x)) - (1-y)log(1-h(x))
        //(-1/m) Sum(Cost)
        final int    NUM_EXAMPLES   = 100; //M
        final int    NUM_PARAMS     = 2; //N
        final double ALPHA          = .001;
        final int    NUM_ITERATIONS = 100000;
        final int    PRINT_AT       = 1000;


        //rows,columns
        //Xi
        //These are the example data, i(down the column) is instance, j is each feature (across the row)
        DoubleMatrix2D independent      = new DenseDoubleMatrix2D(NUM_EXAMPLES,NUM_PARAMS+1);

        //Yi
        //These are the results of the example linear equation.
        DoubleMatrix1D dependent        = new DenseDoubleMatrix1D(NUM_EXAMPLES);
        initLogisticTest( independent, dependent );


        Glm glm = new Glm(independent,dependent,ALPHA,true);
        glm.scaleInputs();

        for( int x=0;x<NUM_ITERATIONS;x++) {
            //thetas = logisticDescent( ALPHA, thetas, independent, dependent );
            glm.step();
            if( x%PRINT_AT == 0) {
                System.out.println(glm.getThetas());
                System.out.println(glm.getCost());
            }
        }

        //It seems like if we don't regularize to Zero mean, then the learning rate has to go way up or it goes off the
        //rails real quick.


        assertEquals(1.7184494794192797,glm.getThetas().get(0), EPSILON);
        assertEquals(4.012902517515474,glm.getThetas().get(1), EPSILON);
        assertEquals(3.743903039594484,glm.getThetas().get(2), EPSILON);

        assertEquals(.7762906907271161,glm.predict(45,85), EPSILON);
    }



    @Test
    public void testClassify() {
        //System.out.println(evalLogistic( new double[]{25.1613,-0.2062,-0.2015}, new double[] {1,45,85} ));
        System.out.println(evalLogistic(new double[]{-25.16133356066622, 0.20623171324620174, 0.20147160039363188}, new double[]{1, 45, 85}));
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

        //Test in place matrix X vector
        SeqBlas.seqBlas.dgemv(false,1.0,examples,thetas,0,result);

        assertEquals(result, new DenseDoubleMatrix1D(new double[]{6,
                                                                  20,
                                                                  7}));

    }
    /**
     * Just for making test data
     */
    private static void add (DoubleMatrix2D independent, DoubleMatrix1D dependent, AtomicInteger cntr, double x1, double x2, int y) {
        independent.set(cntr.get(), 0, 1); //We always set this to 1 for the intercept
        independent.set(cntr.get(), 1, x1);
        independent.set(cntr.get(), 2, x2 );

        dependent.set(cntr.get(), y );

        cntr.incrementAndGet();
    }

    private static void initLogisticTest(DoubleMatrix2D independent,DoubleMatrix1D dependent) {
        AtomicInteger cntr = new AtomicInteger();

        add(independent,dependent,cntr,34.62365962451697,78.0246928153624,0);
        add(independent,dependent,cntr,30.28671076822607,43.89499752400101,0);
        add(independent,dependent,cntr,35.84740876993872,72.90219802708364,0);
        add(independent,dependent,cntr,60.18259938620976,86.30855209546826,1);
        add(independent,dependent,cntr,79.0327360507101,75.3443764369103,1);
        add(independent,dependent,cntr,45.08327747668339,56.3163717815305,0);
        add(independent,dependent,cntr,61.10666453684766,96.51142588489624,1);
        add(independent,dependent,cntr,75.02474556738889,46.55401354116538,1);
        add(independent,dependent,cntr,76.09878670226257,87.42056971926803,1);
        add(independent,dependent,cntr,84.43281996120035,43.53339331072109,1);
        add(independent,dependent,cntr,95.86155507093572,38.22527805795094,0);
        add(independent,dependent,cntr,75.01365838958247,30.60326323428011,0);
        add(independent,dependent,cntr,82.30705337399482,76.48196330235604,1);
        add(independent,dependent,cntr,69.36458875970939,97.71869196188608,1);
        add(independent,dependent,cntr,39.53833914367223,76.03681085115882,0);
        add(independent,dependent,cntr,53.9710521485623,89.20735013750205,1);
        add(independent,dependent,cntr,69.07014406283025,52.74046973016765,1);
        add(independent,dependent,cntr,67.94685547711617,46.67857410673128,0);
        add(independent,dependent,cntr,70.66150955499435,92.92713789364831,1);
        add(independent,dependent,cntr,76.97878372747498,47.57596364975532,1);
        add(independent,dependent,cntr,67.37202754570876,42.83843832029179,0);
        add(independent,dependent,cntr,89.67677575072079,65.79936592745237,1);
        add(independent,dependent,cntr,50.534788289883,48.85581152764205,0);
        add(independent,dependent,cntr,34.21206097786789,44.20952859866288,0);
        add(independent,dependent,cntr,77.9240914545704,68.9723599933059,1);
        add(independent,dependent,cntr,62.27101367004632,69.95445795447587,1);
        add(independent,dependent,cntr,80.1901807509566,44.82162893218353,1);
        add(independent,dependent,cntr,93.114388797442,38.80067033713209,0);
        add(independent,dependent,cntr,61.83020602312595,50.25610789244621,0);
        add(independent,dependent,cntr,38.78580379679423,64.99568095539578,0);
        add(independent,dependent,cntr,61.379289447425,72.80788731317097,1);
        add(independent,dependent,cntr,85.40451939411645,57.05198397627122,1);
        add(independent,dependent,cntr,52.10797973193984,63.12762376881715,0);
        add(independent,dependent,cntr,52.04540476831827,69.43286012045222,1);
        add(independent,dependent,cntr,40.23689373545111,71.16774802184875,0);
        add(independent,dependent,cntr,54.63510555424817,52.21388588061123,0);
        add(independent,dependent,cntr,33.91550010906887,98.86943574220611,0);
        add(independent,dependent,cntr,64.17698887494485,80.90806058670817,1);
        add(independent,dependent,cntr,74.78925295941542,41.57341522824434,0);
        add(independent,dependent,cntr,34.1836400264419,75.2377203360134,0);
        add(independent,dependent,cntr,83.90239366249155,56.30804621605327,1);
        add(independent,dependent,cntr,51.54772026906181,46.85629026349976,0);
        add(independent,dependent,cntr,94.44336776917852,65.56892160559052,1);
        add(independent,dependent,cntr,82.36875375713919,40.61825515970618,0);
        add(independent,dependent,cntr,51.04775177128865,45.82270145776001,0);
        add(independent,dependent,cntr,62.22267576120188,52.06099194836679,0);
        add(independent,dependent,cntr,77.19303492601364,70.45820000180959,1);
        add(independent,dependent,cntr,97.77159928000232,86.7278223300282,1);
        add(independent,dependent,cntr,62.07306379667647,96.76882412413983,1);
        add(independent,dependent,cntr,91.56497449807442,88.69629254546599,1);
        add(independent,dependent,cntr,79.94481794066932,74.16311935043758,1);
        add(independent,dependent,cntr,99.2725269292572,60.99903099844988,1);
        add(independent,dependent,cntr,90.54671411399852,43.39060180650027,1);
        add(independent,dependent,cntr,34.52451385320009,60.39634245837173,0);
        add(independent,dependent,cntr,50.2864961189907,49.80453881323059,0);
        add(independent,dependent,cntr,49.58667721632031,59.80895099453265,0);
        add(independent,dependent,cntr,97.64563396007767,68.86157272420604,1);
        add(independent,dependent,cntr,32.57720016809309,95.59854761387875,0);
        add(independent,dependent,cntr,74.24869136721598,69.82457122657193,1);
        add(independent,dependent,cntr,71.79646205863379,78.45356224515052,1);
        add(independent,dependent,cntr,75.3956114656803,85.75993667331619,1);
        add(independent,dependent,cntr,35.28611281526193,47.02051394723416,0);
        add(independent,dependent,cntr,56.25381749711624,39.26147251058019,0);
        add(independent,dependent,cntr,30.05882244669796,49.59297386723685,0);
        add(independent,dependent,cntr,44.66826172480893,66.45008614558913,0);
        add(independent,dependent,cntr,66.56089447242954,41.09209807936973,0);
        add(independent,dependent,cntr,40.45755098375164,97.53518548909936,1);
        add(independent,dependent,cntr,49.07256321908844,51.88321182073966,0);
        add(independent,dependent,cntr,80.27957401466998,92.11606081344084,1);
        add(independent,dependent,cntr,66.74671856944039,60.99139402740988,1);
        add(independent,dependent,cntr,32.72283304060323,43.30717306430063,0);
        add(independent,dependent,cntr,64.0393204150601,78.03168802018232,1);
        add(independent,dependent,cntr,72.34649422579923,96.22759296761404,1);
        add(independent,dependent,cntr,60.45788573918959,73.09499809758037,1);
        add(independent,dependent,cntr,58.84095621726802,75.85844831279042,1);
        add(independent,dependent,cntr,99.82785779692128,72.36925193383885,1);
        add(independent,dependent,cntr,47.26426910848174,88.47586499559782,1);
        add(independent,dependent,cntr,50.45815980285988,75.80985952982456,1);
        add(independent,dependent,cntr,60.45555629271532,42.50840943572217,0);
        add(independent,dependent,cntr,82.22666157785568,42.71987853716458,0);
        add(independent,dependent,cntr,88.9138964166533,69.80378889835472,1);
        add(independent,dependent,cntr,94.83450672430196,45.69430680250754,1);
        add(independent,dependent,cntr,67.31925746917527,66.58935317747915,1);
        add(independent,dependent,cntr,57.23870631569862,59.51428198012956,1);
        add(independent,dependent,cntr,80.36675600171273,90.96014789746954,1);
        add(independent,dependent,cntr,68.46852178591112,85.59430710452014,1);
        add(independent,dependent,cntr,42.0754545384731,78.84478600148043,0);
        add(independent,dependent,cntr,75.47770200533905,90.42453899753964,1);
        add(independent,dependent,cntr,78.63542434898018,96.64742716885644,1);
        add(independent,dependent,cntr,52.34800398794107,60.76950525602592,0);
        add(independent,dependent,cntr,94.09433112516793,77.15910509073893,1);
        add(independent,dependent,cntr,90.44855097096364,87.50879176484702,1);
        add(independent,dependent,cntr,55.48216114069585,35.57070347228866,0);
        add(independent,dependent,cntr,74.49269241843041,84.84513684930135,1);
        add(independent,dependent,cntr,89.84580670720979,45.35828361091658,1);
        add(independent,dependent,cntr,83.48916274498238,48.38028579728175,1);
        add(independent,dependent,cntr,42.2617008099817,87.10385094025457,1);
        add(independent,dependent,cntr,99.31500880510394,68.77540947206617,1);
        add(independent,dependent,cntr,55.34001756003703,64.9319380069486,1);
        add(independent,dependent,cntr,74.77589300092767,89.52981289513276,1);
    }

}
