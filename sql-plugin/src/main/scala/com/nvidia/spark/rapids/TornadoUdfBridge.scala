/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nvidia.spark.rapids

import java.util.Collections
import java.util.{IdentityHashMap, Map => JMap}

import com.nvidia.spark.RapidsUDF

import org.apache.spark.internal.Logging
import org.apache.spark.sql.catalyst.expressions.ScalaUDF
import org.apache.spark.sql.types.DataType

/**
 * Offers a UDF this plugin's own compiler declined to a GPU kernel compiler, if one is on the
 * classpath and the user asked for it.
 *
 * Why here. `udf-compiler` turns a JVM UDF into Catalyst expressions or gives up; when it gives up
 * the UDF goes to the CPU, even when it is plain numeric arithmetic whose only problem is that it
 * contains a loop, which an expression tree cannot hold. A compiler that emits GPU code rather
 * than expressions has no such problem. This is the one place that asks whether such a compiler
 * wants the UDF, and it asks at planning time, on the driver, because by the time a task is running
 * it is far too late to change the plan.
 *
 * What it does not do. It does not divert every rejected UDF: a UDF is diverted only when the
 * provider positively establishes that it can compile it, and the provider is expected to refuse
 * anything it has not fully understood. It does not reach for a provider unless
 * `spark.rapids.sql.tornadovm.enabled` is set, so an existing deployment behaves exactly as it did.
 * And it does not fail a query: a provider that throws, or is absent, leaves the UDF where it was.
 *
 * The provider is called reflectively so that this plugin does not depend on it to build.
 */
object TornadoUdfBridge extends Logging {

  /** The config that turns this on. Off by default: nothing changes until it is set. */
  val ENABLED_KEY = "spark.rapids.sql.tornadovm.enabled"

  /** Class name of the provider, overridable for testing a different implementation. */
  val PROVIDER_KEY = "spark.rapids.sql.tornadovm.provider"

  private val DEFAULT_PROVIDER = "uk.ac.manchester.tornado.spark.udf.TornadoUdfProvider"

  /** What was decided for each UDF seen in this JVM, for explain output and for tests. */
  private val decisions = Collections.synchronizedMap(new java.util.LinkedHashMap[String, String]())

  /**
   * Why the UDF compiler declined each `ScalaUDF.function`, by identity.
   *
   * The compiler runs during logical-plan resolution and this bridge runs during physical
   * planning, so the reason has to survive in between. Identity is the right key: two UDFs can be
   * equal and still be different registrations, and the function object is what both phases hold.
   */
  private val rejections: JMap[AnyRef, (Array[String], String)] =
    Collections.synchronizedMap(new IdentityHashMap[AnyRef, (Array[String], String)]())

  /** Records that the UDF compiler declined `function`, with the reasons it ran into. */
  def recordRejection(function: AnyRef, reasonNames: Array[String], diagnostics: String): Unit = {
    if (function != null) {
      rejections.put(function, (reasonNames, diagnostics))
    }
  }

  /** Records that the UDF compiler accepted `function`; it keeps its existing path. */
  def recordCompiled(function: AnyRef, udfName: String): Unit = {
    if (function != null) {
      rejections.remove(function)
    }
    decisions.put(udfName, "RAPIDS_COMPILED")
  }

  def isEnabled(conf: RapidsConf): Boolean =
    conf.getStr(ENABLED_KEY).exists(_.trim.equalsIgnoreCase("true"))

  /** The decisions taken so far, newest last. */
  def decisionLog: Map[String, String] = {
    decisions.synchronized {
      import scala.jdk.CollectionConverters._
      decisions.asScala.toMap
    }
  }

  def clearDecisions(): Unit = {
    decisions.clear()
    rejections.clear()
  }

  /**
   * Asks the provider whether it can run `expr` on the GPU.
   *
   * @return a `RapidsUDF` to put in the plan, or `None` to leave the UDF alone.
   */
  def tryCompile(expr: ScalaUDF, conf: RapidsConf): Option[RapidsUDF] = {
    val udfName = expr.udfName.getOrElse("UDF")
    if (!isEnabled(conf)) {
      return None
    }
    val function = expr.function.asInstanceOf[AnyRef]
    val (reasonNames, diagnostics) = Option(rejections.get(function))
      .getOrElse((Array.empty[String], ""))

    try {
      val providerClass = Class.forName(
        conf.getStr(PROVIDER_KEY).getOrElse(DEFAULT_PROVIDER),
        true,
        Thread.currentThread().getContextClassLoader)
      val method = providerClass.getMethod("tryCompile",
        classOf[AnyRef],
        classOf[Array[DataType]],
        classOf[DataType],
        java.lang.Boolean.TYPE,
        java.lang.Boolean.TYPE,
        classOf[String],
        classOf[Array[String]],
        classOf[String])

      val inputTypes = expr.children.map(_.dataType).toArray
      val inputsNullable = expr.children.exists(_.nullable)
      val result = method.invoke(null,
        function,
        inputTypes,
        expr.dataType,
        java.lang.Boolean.valueOf(inputsNullable),
        java.lang.Boolean.valueOf(expr.udfDeterministic),
        udfName,
        reasonNames,
        diagnostics)

      result match {
        case udf: RapidsUDF =>
          decisions.put(udfName, s"TORNADOVM_ELIGIBLE (RAPIDS reasons: ${reasonNames.mkString(",")})")
          logInfo(s"TORNADOVM_ELIGIBLE: $udfName will run on the GPU through $udf")
          Some(udf)
        case _ =>
          decisions.put(udfName, s"TORNADOVM_INELIGIBLE (RAPIDS reasons: ${reasonNames.mkString(",")})")
          logInfo(s"TORNADOVM_INELIGIBLE: $udfName stays on its existing path")
          None
      }
    } catch {
      case e: ClassNotFoundException =>
        decisions.put(udfName, "TORNADOVM_INELIGIBLE (no provider on the classpath)")
        logWarning(s"$ENABLED_KEY is set but no TornadoVM provider is on the classpath: $e")
        None
      case e: Throwable =>
        // A provider that misbehaves must not take the query with it.
        decisions.put(udfName, s"TORNADOVM_INELIGIBLE (provider failed: $e)")
        logWarning(s"The TornadoVM provider failed for $udfName; leaving it on its existing path", e)
        None
    }
  }
}
