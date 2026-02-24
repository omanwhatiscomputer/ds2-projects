package scalation.mathstat

import java.lang.foreign.*
import java.lang.invoke.MethodHandle
import java.lang.foreign.ValueLayout.*

object CudaVectorOps:

  // Load the shared libraries (adjust paths if needed)
  private val libCheck   = SymbolLookup.libraryLookup("src/main/scala/scalation/mathstat/libC/libcudacheck.so", Arena.global())
  private val libKernels = SymbolLookup.libraryLookup("src/main/scala/scalation/mathstat/libC/libcudakernels.so", Arena.global())

  // Get the linker (used to create downcall handles)
  private val linker = Linker.nativeLinker()

  // ---- CUDA availability check (once) ----
  private val cudaIsAvailableAddr: MemorySegment =
    libCheck.find("cuda_is_available")
      .orElseThrow(() => new RuntimeException("Cannot find cuda_is_available"))

  private val cudaIsAvailable: MethodHandle =
    linker.downcallHandle(
      cudaIsAvailableAddr,
      FunctionDescriptor.of(ValueLayout.JAVA_BOOLEAN) // returns bool
    )

  private val _available: Boolean =
    try
      // Use invoke() instead of invokeExact() to allow boxing/unboxing
      val avail = cudaIsAvailable.invoke().asInstanceOf[Boolean]
//      println(s"[CudaVectorOps] CUDA available check returned: $avail")
      avail
    catch case e: Throwable =>
      println(s"[CudaVectorOps] CUDA availability check threw: ${e.getMessage}")
      e.printStackTrace()
      false

  def isAvailable: Boolean = _available

  // ---- Generic helper to invoke a kernel that takes two input arrays and one output array ----
  private def invokeKernel(
                            kernelAddr: MemorySegment,
                            a: Array[Double],
                            b: Array[Double],
                            result: Array[Double],
                            n: Int
                          ): Unit =
    // Create a downcall handle for the kernel (void (double*, double*, double*, int))
    val kernelHandle = linker.downcallHandle(
      kernelAddr,
      FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, ADDRESS, JAVA_INT)
    )

    val byteSize = n * 8L
    // Use a confined arena that will be closed automatically at the end of the block
    val arena = Arena.ofConfined()
    try
      val segA   = arena.allocate(byteSize)
      val segB   = arena.allocate(byteSize)
      val segRes = arena.allocate(byteSize)

      // Copy Java arrays into off‑heap memory
      MemorySegment.copy(a, 0, segA, ValueLayout.JAVA_DOUBLE, 0, n)
      MemorySegment.copy(b, 0, segB, ValueLayout.JAVA_DOUBLE, 0, n)

      // Invoke the native function – use invoke() instead of invokeExact to avoid type mismatch
//      println(s"[CudaVectorOps] Calling native kernel with n=$n")
      kernelHandle.invoke(segA, segB, segRes, n)   // <-- changed to invoke()
//      println("[CudaVectorOps] Native kernel returned")

      // Copy result back into the Java array
      MemorySegment.copy(segRes, ValueLayout.JAVA_DOUBLE, 0, result, 0, n)
    finally
      arena.close()
    end try
  end invokeKernel

  // ---- Addresses for each kernel ----
  private val addAddr: MemorySegment =
    libKernels.find("gpuVectorAdd")
      .orElseThrow(() => new RuntimeException("Cannot find gpuVectorAdd"))

  // ---- Public API for each operation ----
  def add(a: Array[Double], b: Array[Double]): Option[Array[Double]] =
//    println(s"[CudaVectorOps] add called with a.length=${a.length}, b.length=${b.length}")
    if !isAvailable then
      println("[CudaVectorOps] CUDA not available, returning None")
      return None

    val n = a.length
    if n != b.length then
      println(s"[CudaVectorOps] Array length mismatch: a=$n, b=${b.length}, returning None")
      return None

    val result = new Array[Double](n)
    try
//      println("[CudaVectorOps] About to invoke kernel...")
      invokeKernel(addAddr, a, b, result, n)
//      println("[CudaVectorOps] Kernel invocation successful, returning Some(result)")
      Some(result)
    catch case e: Throwable =>
//      println("[CudaVectorOps] Kernel invocation threw exception:")
      e.printStackTrace()
      None

  // Example for future operations:
  // def sub(a: Array[Double], b: Array[Double]): Option[Array[Double]] = ...
  // def mul(a: Array[Double], b: Array[Double]): Option[Array[Double]] = ...
end CudaVectorOps