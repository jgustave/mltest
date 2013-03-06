package com.jd.mltest;

import cern.colt.function.DoubleFunction;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.linalg.Algebra;
import cern.colt.matrix.linalg.SeqBlas;
import cern.jet.math.Functions;

/**
 *
 */
public class Glm {

    private final Algebra        algebra        = new Algebra();




    private final DoubleMatrix2D independent;
    private final DoubleMatrix1D dependent;
    private final DoubleMatrix1D thetas;

    //Size M(num examples). (h)  which is independent * theta
    private final DoubleMatrix1D hypothesies;

    //The transpose points at the same data, just knows it's the transpose.
    private final DoubleMatrix2D independentTransposed;

    //changes to apply to theta
    private final DoubleMatrix1D deltas;


    //learning rate
    @SuppressWarnings ({"FieldCanBeLocal", "UnusedDeclaration"})
    private final double         alpha;

    //alpha * (1/m) in one or just alpha.
    private final double         modifier;

    //Are we doing logistic regression or linear regression
    private final boolean        isLogistic;

    public Glm (DoubleMatrix2D independent,
                DoubleMatrix1D dependent,
                double         alpha,
                boolean        isLogistic ) {


        this.isLogistic             = isLogistic;
        this.alpha                  = alpha;

        this.independent            = independent;
        this.dependent              = dependent;

        this.thetas                 = new DenseDoubleMatrix1D(independent.columns());
        this.hypothesies            = new DenseDoubleMatrix1D(dependent.size());

        this.independentTransposed  = algebra.transpose(independent);
        this.deltas                 = new DenseDoubleMatrix1D(thetas.size());

        for( int x=0;x<thetas.size();x++) {
            thetas.set(x,1);
        }

        if( this.isLogistic ) {
            this.modifier    = alpha;
        }else {
            this.modifier    = alpha / (double)independent.rows();
        }
    }

    /**
     * (h - y)
     */
    private void calcHypothesisError() {

        //In Place
        SeqBlas.seqBlas.dgemv(false,1.0,independent,thetas,0,hypothesies);

        //hypothesies = algebra.mult( independent, thetas );

        if( isLogistic ) {
            hypothesies.assign(new DoubleFunction() {
                @Override
                public double apply (double val) {
                    return( logit( val ) );
                }
            });
        }

        hypothesies.assign(dependent, Functions.minus);
    }

    /**
     * Return the current cost function
     * @return
     */
    public double getCost() {
        if( isLogistic ) {

            //Sum for all samples M
            //(-1/m) * SUM(  (Y * ln(h)) + ((1-Y) * ln(1-h))  )

            //two cross products. gives two scalars. add them. mult by (-1/m)
                //Y * ln(h)
                //(1-Y) * ln(1-h)


            return(0);
        }else {
            //(1/2m) * Sum( (h - y)^2 )
            //sum for all m examples

            //new matrix.. sum of squares of values
            double sumSq = algebra.mult(hypothesies,hypothesies);

            //h is
            return( (1.0/(2.0*hypothesies.size())) * sumSq );
        }
    }

    public DoubleMatrix1D getThetas () {
        return thetas;
    }

    public void step() {
        calcHypothesisError();

        SeqBlas.seqBlas.dgemv(false,1.0,independentTransposed,hypothesies,0,deltas);

//        DoubleMatrix2D transposed = algebra.transpose(independent);
//
//        DoubleMatrix1D deltas     = algebra.mult(transposed, hypothesies );

        // thetas = thetas - (deltas*modifier)  in one step
        thetas.assign(deltas, Functions.minusMult(modifier));
    }

    public static double logit( double val ) {
        return( 1.0 / (1.0 + Math.exp(-val)));
    }

}
