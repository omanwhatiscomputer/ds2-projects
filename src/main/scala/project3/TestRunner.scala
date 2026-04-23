package project3

import scalation.mathstat.{VectorD}

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

  // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
}