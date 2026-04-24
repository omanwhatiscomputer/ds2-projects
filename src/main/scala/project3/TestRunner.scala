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
}