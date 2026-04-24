package project3

import scalation.mathstat.{VectorD, MatrixD}

@main def TestRunner(): Unit = {
  println("Hello World");
  val x = VectorD (2.0, 1.0, 2.0);
  val y = VectorD (2.0, 1.0, 2.0);
  

  val z = x+y;
  val a = x-y;
  val b = x*y;
  val c = x/y;

  println(z);
  println(a);
  println(b);
  println(c);

  // %%%%%%%%%%%%%%%%%%%%%%%%%%%

  val d = x + 1;
  val e = x - 2;
  val f = x * 2;
  val g = x / 2;

  println(d);
  println(e);
  println(f);
  println(g);

  // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  // MatrixD tests (element-wise matrix op matrix — 4 cases)
  val m1 = MatrixD((2, 2), 1.0, 2.0, 3.0, 4.0)
  val m2 = MatrixD((2, 2), 5.0, 6.0, 7.0, 8.0)

  val m3 = m1 + m2
  val m4 = m1 - m2
  val m5 = m1 *~ m2
  val m6 = m1 / m2

  println(m3)
  println(m4)
  println(m5)
  println(m6)

  // MatrixD tests (matrix op scalar — 4 cases)
  val m7 = m1 + 10.0
  val m8 = m1 - 1.0
  val m9 = m1 * 2.0
  val m10 = m1 / 2.0

  println(m7)
  println(m8)
  println(m9)
  println(m10)

  // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  // MatrixD tests (matrix op row vector — broadcast across rows)
  // m1 = [[1,2],[3,4]], rowVec = [10,20]
  // +(rowVec): [[11,22],[13,24]]
  // -(rowVec): [[-9,-18],[-7,-16]]
  // *~(rowVec): [[10,40],[30,80]]
  // /(rowVec): [[0.1,0.1],[0.3,0.2]]
  val rowVec = VectorD(10.0, 20.0)

  val mr1 = m1 + rowVec
  val mr2 = m1 - rowVec
  val mr3 = m1 *~ rowVec
  val mr4 = m1 / rowVec

  println(mr1)
  println(mr2)
  println(mr3)
  println(mr4)

  // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  // MatrixD tests (matrix op col vector — broadcast across cols)
  // m1 = [[1,2],[3,4]], colVec = [100,200]
  // +^(colVec): [[101,102],[203,204]]
  // -^(colVec): [[-99,-98],[-197,-196]]
  // *~:(colVec): [[100,200],[600,800]]
  val colVec = VectorD(100.0, 200.0)

  val mc1 = m1 +^ colVec
  val mc2 = m1 -^ colVec
  val mc3 = colVec *~: m1

  println(mc1)
  println(mc2)
  println(mc3)

  // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  // MatrixD tests: matrix multiplication (*(MatrixD), mul)
  // m1 = [[1,2],[3,4]], m2 = [[5,6],[7,8]]
  // m1 * m2  = [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
  // m1 mul m2 = same result
  val mm1 = m1 * m2
  val mm2 = m1 mul m2
  println(mm1)
  println(mm2)

  // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  // MatrixD tests: matrix * vector (*(VectorD))
  // m1 = [[1,2],[3,4]], v = [10,20]
  // m1 * v = [1*10+2*20, 3*10+4*20] = [50, 110]
  val mv1 = m1 * rowVec
  println(mv1)

  // MatrixD tests: transposed GEMV (*:, dot(VectorD))
  // m1^T = [[1,3],[2,4]], colVec = [100,200]
  // colVec *: m1 = m1^T * colVec = [1*100+3*200, 2*100+4*200] = [700, 1000]
  // m1 dot colVec = same result (A^T * v)
  val mv2 = colVec *: m1
  val mv3 = m1 dot colVec
  println(mv2)
  println(mv3)

  // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  // MatrixD tests: reductions
  // m1 = [[1,2],[3,4]]
  // sum   = 10,  mmax = 4,  mmin = 1,  mmean = 2.5
  // sumV  = [4,6],  max = [3,4],  min = [1,2],  mean = [2.0,3.0]
  // sumVr = [3,7]
  println(m1.sum)
  println(m1.mmax)
  println(m1.mmin)
  println(m1.mmean)
  println(m1.sumV)
  println(m1.max)
  println(m1.min)
  println(m1.mean)
  println(m1.sumVr)

  // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
}