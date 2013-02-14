package com.jd.mltest;

import cern.colt.matrix.DoubleMatrix2D;
import org.junit.Test;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;

/**
 *
 */
public class TestNothing {

    @Test
    public void testNothing() {
        DoubleMatrix2D helloMatrix = new DenseDoubleMatrix2D(4,5);
        helloMatrix.assign(1D);
    }

}
