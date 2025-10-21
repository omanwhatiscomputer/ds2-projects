//package project1
//import java.lang.foreign.{Arena, MemorySegment, ValueLayout, Linker, FunctionDescriptor, SymbolLookup}
//import scala.util.Using
//import java.lang.invoke.MethodHandle
//
//@main def HelloWorld(): Unit = {
//
////  Using.resource(Arena.ofConfined()) { arena =>
////    val nativeText: MemorySegment = arena.allocateFrom(s)
////
////    var i = 0
////    while (i < s.length) do {
////      // Note: offset is a long
////      val byte = nativeText.get(ValueLayout.JAVA_BYTE, i.toLong)
////      print(byte.toChar)
////      i += 1
////    }
////  } // arena.close() is called automatically
//  val linker: Linker = Linker.nativeLinker()
//  val ADD_SIG: FunctionDescriptor = {
//    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT)
//  }
//  val sum: Int = Using.resource(Arena.ofConfined()) { arena =>
//    val lookup: SymbolLookup = SymbolLookup.libraryLookup("/mnt/c/Libs/scalation_2.0/data/libadd.so", arena)
//    val addr: MemorySegment =
//      lookup.find("add").orElseThrow(() => new UnsatisfiedLinkError("symbol 'add' not found"))
//    val add: MethodHandle = linker.downcallHandle(addr, ADD_SIG)
//
//    // invokeExact requires exact primitive types
//    add.invokeExact(2: Int, 5: Int).asInstanceOf[Int]
//  }
//  println(sum)
//}
