package com.jd.mltest;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.linalg.Algebra;

import cern.colt.matrix.linalg.SeqBlas;
import org.junit.Test;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;


import java.util.Arrays;
import java.util.Random;
import static org.junit.Assert.assertEquals;

/**
 * Just the proof of concept area.. No classes yet.
 */
public class TestSimpleDescent {
    public static final double EPSILON = 0.0001;


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

        Glm glm = new Glm(independent,dependent,ALPHA, false, null );
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
        final double ALPHA          = .00001;
        final int    NUM_ITERATIONS = 100000000;
        final int    PRINT_AT       = 1000000;

        //rows,columns
        //Xi
        //These are the example data, i(down the column) is instance, j is each feature (across the row)
        DoubleMatrix2D independent      = new DenseDoubleMatrix2D(getIndep(getFirstTestData(),true));

        //Yi
        //These are the results of the example linear equation.
        DoubleMatrix1D dependent        = new DenseDoubleMatrix1D(getDep(getFirstTestData()));

        Glm glm = new Glm(independent,dependent,ALPHA,true, null);


        for( int x=0;x<NUM_ITERATIONS;x++) {
            //thetas = logisticDescent( ALPHA, thetas, independent, dependent );
            glm.step();
            if( x%PRINT_AT == 0) {
                System.out.println("Cost:     " + glm.getCost());
                System.out.println("Gradient: " + glm.getGradient());
                System.out.println("Theta:    " + glm.getThetas());
            }
        }


        System.out.println("Cost:     " + glm.getCost());
        System.out.println("Gradient: " + glm.getGradient());
        System.out.println("Theta:    " + glm.getThetas());


        //It seems like if we don't regularize to Zero mean, then the learning rate has to go way up or it goes off the
        //rails real quick.


        assertEquals(-25.16133356066622,glm.getThetas().get(0), EPSILON);
        assertEquals(0.20623171324620174,glm.getThetas().get(1), EPSILON);
        assertEquals(0.20147160039363188, glm.getThetas().get(2), EPSILON);

        assertEquals(.7762906907271161,glm.predict(45,85), EPSILON);
    }


    @Test
    public void testLogisticDescentMultipleTwo() {
        final double ALPHA          = .01;
        final int    NUM_ITERATIONS = 50000000;
        final long   PRINT_AT       = 1000000;

        //rows,columns
        //Xi
        //These are the example data, i(down the column) is instance, j is each feature (across the row)
        DoubleMatrix2D independent      = new DenseDoubleMatrix2D(getIndep(getData2(),true));
        //independent = mapFeature(independent);
        //Yi
        //These are the results of the example linear equation.
        DoubleMatrix1D dependent        = new DenseDoubleMatrix1D(getDep(getData2()));

        Glm glm = new Glm(independent,dependent,ALPHA,true, null);

        for( int x=0;x<NUM_ITERATIONS;x++) {
            glm.step();

            long test = x/PRINT_AT;
            if( x%PRINT_AT == 0) {
                System.out.println("TEST:     " + test + " " + glm.getAlpha() );
                System.out.println("Cost:     " + glm.getCost());
                System.out.println("Gradient: " + glm.getGradient());
                System.out.println("Theta:    " + Arrays.toString(glm.getThetas().toArray()));
            }

            if( test > 40 ) {
                glm.setAlpha(.00000000001);
            }else if( test > 20 ) {
                glm.setAlpha(.000000001);
            }else if(test > 10 ) {
                glm.setAlpha(.000001);
            }else if(test > 5 ) {
                glm.setAlpha(.00001);
            }else if(test > 2 ) {
                glm.setAlpha(.0001);
            }

            x++;
        }


        System.out.println("Cost:     " + glm.getCost());
        System.out.println("Gradient: " + glm.getGradient());
        System.out.println("Theta:    " + glm.getThetas());

//        Gradient: 1 x 3 matrix
//        6.849512E-016 -2.352073E-014 -1.469634E-015
//        Theta:    1 x 3 matrix
//        -0.014184 -0.303521 -0.018132
//        Cost:     0.6902411220169706

    }

    @Test
    /**
     * Now we have feature scaled and get the same results.
     */
    public void testLogisticDescentMultipleScaled() {
        //Cost function: -y * log(h(x)) - (1-y)log(1-h(x))
        //(-1/m) Sum(Cost)
        final double ALPHA          = .001;
        final int    NUM_ITERATIONS = 100000;
        final int    PRINT_AT       = 1000;


        //rows,columns
        //Xi
        //These are the example data, i(down the column) is instance, j is each feature (across the row)
        DoubleMatrix2D independent      = new DenseDoubleMatrix2D(getIndep(getFirstTestData(),true));

        //Yi
        //These are the results of the example linear equation.
        DoubleMatrix1D dependent        = new DenseDoubleMatrix1D(getDep(getFirstTestData()));


        Glm glm = new Glm(independent,dependent,ALPHA,true, null);
        glm.scaleInputs();

        for( int x=0;x<NUM_ITERATIONS;x++) {
            //thetas = logisticDescent( ALPHA, thetas, independent, dependent );
            glm.step();
            if( x%PRINT_AT == 0) {
                System.out.println("Cost:     " + glm.getCost());
                System.out.println("Gradient: " + glm.getGradient());
                System.out.println("Theta:    " + glm.getThetas());
            }
        }

        //It seems like if we don't regularize to Zero mean, then the learning rate has to go way up or it goes off the
        //rails real quick.

        System.out.println("Cost:     " + glm.getCost());
        System.out.println("Gradient: " + glm.getGradient());
        System.out.println("Theta:    " + glm.getThetas());


        assertEquals(1.7184494794192797,glm.getThetas().get(0), EPSILON);
        assertEquals(4.012902517515474,glm.getThetas().get(1), EPSILON);
        assertEquals(3.743903039594484,glm.getThetas().get(2), EPSILON);

        assertEquals(.7762906907271161,glm.predict(45,85), EPSILON);
    }



    @Test
    /**
     * Now we have feature scaled and get the same results.
     */
    public void testLogisticDescentScaledAndRegular() {
        //Cost function: -y * log(h(x)) - (1-y)log(1-h(x))
        //(-1/m) Sum(Cost)

        final double ALPHA          = .001;
        final int    NUM_ITERATIONS = 10000000;
        final int    PRINT_AT       = 100000;


        //rows,columns
        //Xi
        //These are the example data, i(down the column) is instance, j is each feature (across the row)
        DoubleMatrix2D independent      = new DenseDoubleMatrix2D(getIndep(getData2(), true));

        //Yi
        //These are the results of the example linear equation.
        DoubleMatrix1D dependent        = new DenseDoubleMatrix1D(getDep(getData2()));


        //Add all sort so interacitons
        //TODO: I don't think map feature looks correct.. strange repeat values
        independent = mapFeature(independent);


        Glm glm = new Glm(independent,dependent,ALPHA,true, 1.0 );

        //For testing start at all 0.
        glm.getThetas().assign(0);


        //TODO: scaling input gives lots of NANs!
        //glm.scaleInputs();

        for( int x=0;x<NUM_ITERATIONS;x++) {
            //thetas = logisticDescent( ALPHA, thetas, independent, dependent );
            glm.step();
            if( x%PRINT_AT == 0) {
                System.out.println("Cost:     " + glm.getCost());
                System.out.println("Gradient: " + glm.getGradient());
                System.out.println("Theta:    " + glm.getThetas());
            }
        }

        System.out.println("Cost:     " + glm.getCost());
        System.out.println("Gradient: " + glm.getGradient());
        System.out.println("Theta:    " + glm.getThetas());


        //From Test Coursera test data:
        assertEquals(.693, glm.getCost(), EPSILON );

        //It seems like if we don't regularize to Zero mean, then the learning rate has to go way up or it goes off the
        //rails real quick.


//        assertEquals(1.7184494794192797,glm.getThetas().get(0), EPSILON);
//        assertEquals(4.012902517515474,glm.getThetas().get(1), EPSILON);
//        assertEquals(3.743903039594484,glm.getThetas().get(2), EPSILON);

//        assertEquals(.7762906907271161,glm.predict(45,85), EPSILON);
    }


    @Test
    /**
     * Test Cost and Gradient by hand
     */
    public void testRegLogisticCalcs() {
        //2 params (plus Int), 2 instances.
        //Lets verify by hand what these funcs should be

        DoubleMatrix2D independent      = new DenseDoubleMatrix2D(new double[][]{{1.0,1.5,0.2},{1.0,0.3,0.4}});

        //Yi
        //These are the results of the example linear equation.
        DoubleMatrix1D dependent        = new DenseDoubleMatrix1D(new double[]{1.0,0.0});


        double ALPHA  = 0.1;
        double LAMBDA = 1.0;

        Glm glm = new Glm( independent, dependent, ALPHA, true, LAMBDA );

        glm.getThetas().set(0,1.1);
        glm.getThetas().set(1,1.2);
        glm.getThetas().set(2,1.3);

        double cost         = glm.getCost();
        double[] gradient   = glm.getGradient().toArray();

        //COST = ( (-1/m) * SUM( (Y * log(h))  +  ((1-Y)*log(1-h)) ) ) )   +  lambda/(2m) * Sum(theta^2)


        double h1 = 1.0*1.1 + 1.5*1.2 + .2*1.3;
        double h2 = 1.0*1.1 + .3*1.2  + .4*1.3;


        h1 = Glm.logit(h1);
        h2 = Glm.logit(h2);


        double lhs = Math.log(h1);
        double rhs = Math.log(1.0-h2);

        double calcCost = (-1.0/2.0) * (lhs+rhs);
        calcCost += ( (LAMBDA/(2.0*2.0)) * ((1.1*1.1)+  (1.2*1.2) + (1.3*1.3)) );

        assertEquals(cost,calcCost,EPSILON);

        double diff0 = ((h1-1.0)*1.0) + ((h2-0.0)*1.0);
        double diff1 = ((h1-1.0)*1.5) + ((h2-0.0)*0.3);
        double diff2 = ((h1-1.0)*0.2) + ((h2-0.0)*0.4);

        diff0 = diff0 * (1.0/2.0);
        diff1 = diff1 * (1.0/2.0);
        diff2 = diff2 * (1.0/2.0);

        diff1 += (LAMBDA / 2.0) * 1.2;
        diff2 += (LAMBDA / 2.0) * 1.3;


        assertEquals(diff0,gradient[0],EPSILON);
        assertEquals(diff1,gradient[1],EPSILON);
        assertEquals(diff2,gradient[2],EPSILON);

    }

    @Test
    /**
     * Test Cost and Gradient by hand
     */
    public void testRegLogisticCalcsThree() {
        //2 params (plus Int), 2 instances.
        //Lets verify by hand what these funcs should be

        DoubleMatrix2D independent      = new DenseDoubleMatrix2D(new double[][]{{1.0,1.5,0.2},
                                                                                 {1.0,0.3,0.4},
                                                                                 {0.1,0.2,0.3},
                                                                                 {0.4,0.5,0.6}});

        //Yi
        //These are the results of the example linear equation.
        DoubleMatrix1D dependent        = new DenseDoubleMatrix1D(new double[]{1.0,0.0,1.0,0.0});


        double ALPHA  = 0.1;
        double LAMBDA = 0.5;
        double M      = dependent.size();

        Glm glm = new Glm( independent, dependent, ALPHA, true, LAMBDA );

        glm.getThetas().set(0,1.1);
        glm.getThetas().set(1,1.2);
        glm.getThetas().set(2,1.3);

        double cost         = glm.getCost();
        double[] gradient   = glm.getGradient().toArray();

        //COST = ( (-1/m) * SUM( (Y * log(h))  +  ((1-Y)*log(1-h)) ) ) )   +  lambda/(2m) * Sum(theta^2)


        double h1 = 1.0*1.1 + 1.5*1.2 + .2*1.3;
        double h2 = 1.0*1.1 + .3*1.2  + .4*1.3;
        double h3 = .1*1.1 + .2*1.2 + .3*1.3;
        double h4 = .4*1.1 + .5*1.2  + .6*1.3;


        h1 = Glm.logit(h1); //lhs
        h2 = Glm.logit(h2); //rhs
        h3 = Glm.logit(h3); //lhs
        h4 = Glm.logit(h4); //rhs


        double lhs = Math.log(h1)+Math.log(h3);
        double rhs = Math.log(1.0-h2)+Math.log(1.0-h4);

        double calcCost = (-1.0/M) * (lhs+rhs);
        calcCost += ( (LAMBDA/(2.0*M)) * ((1.1*1.1)+  (1.2*1.2) + (1.3*1.3)) );

        assertEquals(cost,calcCost,EPSILON);

        //gradientj = (1/m)*SUM( diff(i) * x(i)j )
        //j=0
        double diff0 = ((h1-1.0)*1.0) + ((h2-0.0)*1.0) + ((h3-1.0)*.1) + ((h4-0.0)*.4);
        //j=1
        double diff1 = ((h1-1.0)*1.5) + ((h2-0.0)*0.3) + ((h3-1.0)*.2) + ((h4-0.0)*.5);
        //j=2
        double diff2 = ((h1-1.0)*0.2) + ((h2-0.0)*0.4) + ((h3-1.0)*.3) + ((h4-0.0)*.6);

        diff0 = diff0 * (1.0/M);
        diff1 = diff1 * (1.0/M);
        diff2 = diff2 * (1.0/M);

        diff1 += (LAMBDA / M) * 1.2;
        diff2 += (LAMBDA / M) * 1.3;


        assertEquals(diff0,gradient[0],EPSILON);
        assertEquals(diff1,gradient[1],EPSILON);
        assertEquals(diff2,gradient[2],EPSILON);

    }
    @Test
    /**
     * Test Cost and Gradient by hand
     */
    public void testRegLogisticCalcsTwo() {
        //2 params (plus Int), 2 instances.
        //Lets verify by hand what these funcs should be

        DoubleMatrix2D independent      = new DenseDoubleMatrix2D(new double[][]{{1.0,1.5,0.2},{1.0,0.3,0.4}});

        //Yi
        //These are the results of the example linear equation.
        DoubleMatrix1D dependent        = new DenseDoubleMatrix1D(new double[]{1.0,1.0});


        double ALPHA  = 0.1;
        double LAMBDA = 1.0;

        Glm glm = new Glm( independent, dependent, ALPHA, true, LAMBDA );

        glm.getThetas().set(0,1.1);
        glm.getThetas().set(1,1.2);
        glm.getThetas().set(2,1.3);

        double cost         = glm.getCost();
        double[] gradient   = glm.getGradient().toArray();

        //COST = ( (-1/m) * SUM( (Y * log(h))  +  ((1-Y)*log(1-h)) ) ) )   +  lambda/(2m) * Sum(theta^2)


        double h1 = 1.0*1.1 + 1.5*1.2 + .2*1.3;
        double h2 = 1.0*1.1 + .3*1.2  + .4*1.3;


        h1 = Glm.logit(h1);
        h2 = Glm.logit(h2);


        double lhs = Math.log(h1)+Math.log(h2);
        double rhs = 0;

        double calcCost = (-1.0/2.0) * (lhs+rhs);
        calcCost += ( (LAMBDA/(2.0*2.0)) * ((1.1*1.1)+(1.2*1.2) + (1.3*1.3)) );

        assertEquals(cost,calcCost,EPSILON);

        //gradientj = (1/m)*SUM( diff(i) * x(i)j )

        //Sum diff (i) * x(j),  per j
        double diff0 = ((h1-1.0)*1.0) + ((h2-1.0)*1.0);
        double diff1 = ((h1-1.0)*1.5) + ((h2-1.0)*0.3);
        double diff2 = ((h1-1.0)*0.2) + ((h2-1.0)*0.4);

        //1/m
        diff0 = diff0 * (1.0/2.0);
        diff1 = diff1 * (1.0/2.0);
        diff2 = diff2 * (1.0/2.0);

        //regularize
        diff1 += (LAMBDA / 2.0) * 1.2;
        diff2 += (LAMBDA / 2.0) * 1.3;


        assertEquals(diff0,gradient[0],EPSILON);
        assertEquals(diff1,gradient[1],EPSILON);
        assertEquals(diff2,gradient[2],EPSILON);

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

    public static double[][] getFirstTestData() {
        double result[][] = {
            {34.62365962451697,78.0246928153624,0},
            {30.28671076822607,43.89499752400101,0},
            {35.84740876993872,72.90219802708364,0},
            {60.18259938620976,86.30855209546826,1},
            {79.0327360507101,75.3443764369103,1},
            {45.08327747668339,56.3163717815305,0},
            {61.10666453684766,96.51142588489624,1},
            {75.02474556738889,46.55401354116538,1},
            {76.09878670226257,87.42056971926803,1},
            {84.43281996120035,43.53339331072109,1},
            {95.86155507093572,38.22527805795094,0},
            {75.01365838958247,30.60326323428011,0},
            {82.30705337399482,76.48196330235604,1},
            {69.36458875970939,97.71869196188608,1},
            {39.53833914367223,76.03681085115882,0},
            {53.9710521485623,89.20735013750205,1},
            {69.07014406283025,52.74046973016765,1},
            {67.94685547711617,46.67857410673128,0},
            {70.66150955499435,92.92713789364831,1},
            {76.97878372747498,47.57596364975532,1},
            {67.37202754570876,42.83843832029179,0},
            {89.67677575072079,65.79936592745237,1},
            {50.534788289883,48.85581152764205,0},
            {34.21206097786789,44.20952859866288,0},
            {77.9240914545704,68.9723599933059,1},
            {62.27101367004632,69.95445795447587,1},
            {80.1901807509566,44.82162893218353,1},
            {93.114388797442,38.80067033713209,0},
            {61.83020602312595,50.25610789244621,0},
            {38.78580379679423,64.99568095539578,0},
            {61.379289447425,72.80788731317097,1},
            {85.40451939411645,57.05198397627122,1},
            {52.10797973193984,63.12762376881715,0},
            {52.04540476831827,69.43286012045222,1},
            {40.23689373545111,71.16774802184875,0},
            {54.63510555424817,52.21388588061123,0},
            {33.91550010906887,98.86943574220611,0},
            {64.17698887494485,80.90806058670817,1},
            {74.78925295941542,41.57341522824434,0},
            {34.1836400264419,75.2377203360134,0},
            {83.90239366249155,56.30804621605327,1},
            {51.54772026906181,46.85629026349976,0},
            {94.44336776917852,65.56892160559052,1},
            {82.36875375713919,40.61825515970618,0},
            {51.04775177128865,45.82270145776001,0},
            {62.22267576120188,52.06099194836679,0},
            {77.19303492601364,70.45820000180959,1},
            {97.77159928000232,86.7278223300282,1},
            {62.07306379667647,96.76882412413983,1},
            {91.56497449807442,88.69629254546599,1},
            {79.94481794066932,74.16311935043758,1},
            {99.2725269292572,60.99903099844988,1},
            {90.54671411399852,43.39060180650027,1},
            {34.52451385320009,60.39634245837173,0},
            {50.2864961189907,49.80453881323059,0},
            {49.58667721632031,59.80895099453265,0},
            {97.64563396007767,68.86157272420604,1},
            {32.57720016809309,95.59854761387875,0},
            {74.24869136721598,69.82457122657193,1},
            {71.79646205863379,78.45356224515052,1},
            {75.3956114656803,85.75993667331619,1},
            {35.28611281526193,47.02051394723416,0},
            {56.25381749711624,39.26147251058019,0},
            {30.05882244669796,49.59297386723685,0},
            {44.66826172480893,66.45008614558913,0},
            {66.56089447242954,41.09209807936973,0},
            {40.45755098375164,97.53518548909936,1},
            {49.07256321908844,51.88321182073966,0},
            {80.27957401466998,92.11606081344084,1},
            {66.74671856944039,60.99139402740988,1},
            {32.72283304060323,43.30717306430063,0},
            {64.0393204150601,78.03168802018232,1},
            {72.34649422579923,96.22759296761404,1},
            {60.45788573918959,73.09499809758037,1},
            {58.84095621726802,75.85844831279042,1},
            {99.82785779692128,72.36925193383885,1},
            {47.26426910848174,88.47586499559782,1},
            {50.45815980285988,75.80985952982456,1},
            {60.45555629271532,42.50840943572217,0},
            {82.22666157785568,42.71987853716458,0},
            {88.9138964166533,69.80378889835472,1},
            {94.83450672430196,45.69430680250754,1},
            {67.31925746917527,66.58935317747915,1},
            {57.23870631569862,59.51428198012956,1},
            {80.36675600171273,90.96014789746954,1},
            {68.46852178591112,85.59430710452014,1},
            {42.0754545384731,78.84478600148043,0},
            {75.47770200533905,90.42453899753964,1},
            {78.63542434898018,96.64742716885644,1},
            {52.34800398794107,60.76950525602592,0},
            {94.09433112516793,77.15910509073893,1},
            {90.44855097096364,87.50879176484702,1},
            {55.48216114069585,35.57070347228866,0},
            {74.49269241843041,84.84513684930135,1},
            {89.84580670720979,45.35828361091658,1},
            {83.48916274498238,48.38028579728175,1},
            {42.2617008099817,87.10385094025457,1},
            {99.31500880510394,68.77540947206617,1},
            {55.34001756003703,64.9319380069486,1},
            {74.77589300092767,89.52981289513276,1},
        };
        return( result );
    }
    public static double[][] getIndep(double[][] rawData, boolean addIntercept) {
        int numVars = rawData[0].length-1;
        double[][] result = new double[rawData.length][numVars+(addIntercept?1:0)];
        for( int x=0;x<result.length;x++) {
            if( addIntercept ) {
                result[x] = new double[numVars+1];
                result[x][0]=1;
                System.arraycopy(rawData[x],0,result[x],1,numVars);
            }else {
                result[x] = Arrays.copyOfRange(rawData[x],0,numVars);
            }
        }
        return( result );
    }
    public static double[] getDep(double[][] rawData ) {
        int      depIndex = rawData[0].length-1;
        double[] result   = new double[rawData.length];
        for( int x=0;x<result.length;x++) {
            result[x] = rawData[x][depIndex];
        }
        return( result );
    }

    public static double[][] getData2() {
        double[][] result = new double[][] {
        {0.051267,0.69956,1},
        {-0.092742,0.68494,1},
        {-0.21371,0.69225,1},
        {-0.375,0.50219,1},
        {-0.51325,0.46564,1},
        {-0.52477,0.2098,1},
        {-0.39804,0.034357,1},
        {-0.30588,-0.19225,1},
        {0.016705,-0.40424,1},
        {0.13191,-0.51389,1},
        {0.38537,-0.56506,1},
        {0.52938,-0.5212,1},
        {0.63882,-0.24342,1},
        {0.73675,-0.18494,1},
        {0.54666,0.48757,1},
        {0.322,0.5826,1},
        {0.16647,0.53874,1},
        {-0.046659,0.81652,1},
        {-0.17339,0.69956,1},
        {-0.47869,0.63377,1},
        {-0.60541,0.59722,1},
        {-0.62846,0.33406,1},
        {-0.59389,0.005117,1},
        {-0.42108,-0.27266,1},
        {-0.11578,-0.39693,1},
        {0.20104,-0.60161,1},
        {0.46601,-0.53582,1},
        {0.67339,-0.53582,1},
        {-0.13882,0.54605,1},
        {-0.29435,0.77997,1},
        {-0.26555,0.96272,1},
        {-0.16187,0.8019,1},
        {-0.17339,0.64839,1},
        {-0.28283,0.47295,1},
        {-0.36348,0.31213,1},
        {-0.30012,0.027047,1},
        {-0.23675,-0.21418,1},
        {-0.06394,-0.18494,1},
        {0.062788,-0.16301,1},
        {0.22984,-0.41155,1},
        {0.2932,-0.2288,1},
        {0.48329,-0.18494,1},
        {0.64459,-0.14108,1},
        {0.46025,0.012427,1},
        {0.6273,0.15863,1},
        {0.57546,0.26827,1},
        {0.72523,0.44371,1},
        {0.22408,0.52412,1},
        {0.44297,0.67032,1},
        {0.322,0.69225,1},
        {0.13767,0.57529,1},
        {-0.0063364,0.39985,1},
        {-0.092742,0.55336,1},
        {-0.20795,0.35599,1},
        {-0.20795,0.17325,1},
        {-0.43836,0.21711,1},
        {-0.21947,-0.016813,1},
        {-0.13882,-0.27266,1},
        {0.18376,0.93348,0},
        {0.22408,0.77997,0},
        {0.29896,0.61915,0},
        {0.50634,0.75804,0},
        {0.61578,0.7288,0},
        {0.60426,0.59722,0},
        {0.76555,0.50219,0},
        {0.92684,0.3633,0},
        {0.82316,0.27558,0},
        {0.96141,0.085526,0},
        {0.93836,0.012427,0},
        {0.86348,-0.082602,0},
        {0.89804,-0.20687,0},
        {0.85196,-0.36769,0},
        {0.82892,-0.5212,0},
        {0.79435,-0.55775,0},
        {0.59274,-0.7405,0},
        {0.51786,-0.5943,0},
        {0.46601,-0.41886,0},
        {0.35081,-0.57968,0},
        {0.28744,-0.76974,0},
        {0.085829,-0.75512,0},
        {0.14919,-0.57968,0},
        {-0.13306,-0.4481,0},
        {-0.40956,-0.41155,0},
        {-0.39228,-0.25804,0},
        {-0.74366,-0.25804,0},
        {-0.69758,0.041667,0},
        {-0.75518,0.2902,0},
        {-0.69758,0.68494,0},
        {-0.4038,0.70687,0},
        {-0.38076,0.91886,0},
        {-0.50749,0.90424,0},
        {-0.54781,0.70687,0},
        {0.10311,0.77997,0},
        {0.057028,0.91886,0},
        {-0.10426,0.99196,0},
        {-0.081221,1.1089,0},
        {0.28744,1.087,0},
        {0.39689,0.82383,0},
        {0.63882,0.88962,0},
        {0.82316,0.66301,0},
        {0.67339,0.64108,0},
        {1.0709,0.10015,0},
        {-0.046659,-0.57968,0},
        {-0.23675,-0.63816,0},
        {-0.15035,-0.36769,0},
        {-0.49021,-0.3019,0},
        {-0.46717,-0.13377,0},
        {-0.28859,-0.060673,0},
        {-0.61118,-0.067982,0},
        {-0.66302,-0.21418,0},
        {-0.59965,-0.41886,0},
        {-0.72638,-0.082602,0},
        {-0.83007,0.31213,0},
        {-0.72062,0.53874,0},
        {-0.59389,0.49488,0},
        {-0.48445,0.99927,0},
        {-0.0063364,0.99927,0},
        {0.63265,-0.030612,0}
        };
        return( result );
    }
    /**
     * Takes input matrix, and creates new features.
     * skips first column, but combines 2nd and 3rd in various ways.
     * X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
     * @param input
     * @return
     */
    public static DoubleMatrix2D mapFeature(DoubleMatrix2D input){
        DoubleMatrix2D result = new DenseDoubleMatrix2D(input.rows(),28);
        for( int z=0;z<input.rows();z++) {
            result.setQuick(z,0,1);
        }

        //We are going column by column..

        double subResult = 0;
        int destColumn = 1; //skip intercept
        for( int i=1;i<=6;i++) {
            for( int j=0;j<=i;j++) {
                //int a = i-j;
                if( (i-j) == 0 ) {
                    //System.out.println(" Y^" +j  );
                    for( int z=0;z<input.rows();z++) {
                        double yInput = input.get(z,1);
                        subResult = Math.pow(yInput,j);
                        result.setQuick(z,destColumn,subResult);
                    }
                }
                else if( j == 0 ) {
                    //System.out.println("X^"+(i-j)  );
                    for( int z=0;z<input.rows();z++) {
                        double xInput = input.get(z,0);
                        subResult = Math.pow(xInput,(i-j) );
                        result.setQuick(z,destColumn,subResult);
                    }

                }
                else {
                    for( int z=0;z<input.rows();z++) {
                        double xInput = input.get(z,0);
                        double yInput = input.get(z,1);
                        subResult = Math.pow(xInput,(i-j)) * Math.pow(yInput,j );
                        result.setQuick(z,destColumn,subResult);
                        //System.out.println("X^"+(i-j) + "* Y^" +j  );
                    }
                }

                destColumn++;
            }
        }
        return( result );
    }

    @Test
    public void testFoo() {
        int count = 0;
        for( int i=1;i<=6;i++) {
            for( int j=0;j<=i;j++) {
                int a = i-j;
                if( a == 0 ) {
                    System.out.println(" Y^" +j  );
                }
                else if( j == 0 ) {
                    System.out.println("X^"+(i-j)  );
                }
                else {
                    System.out.println("X^"+(i-j) + "* Y^" +j  );
                }
                count++;
            }
        }
        System.out.println("GOT:" + count);
    }



//    degree = 6;
//    out = ones(size(X1(:,1)));
//    for i = 1:degree
//        for j = 0:i
//            out(:, end+1) = (X1.^(i-j)).*(X2.^j);
//        end
//    end
//
//    end

}
