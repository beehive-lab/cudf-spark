/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

package com.nvidia.spark.udf

import scala.util.control.NonFatal

import com.nvidia.spark.rapids.shims.ShimExpression

import org.apache.spark.SparkException
import org.apache.spark.internal.Logging
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Expression, ScalaUDF}
import org.apache.spark.sql.catalyst.expressions.codegen.{CodegenContext, ExprCode}
import org.apache.spark.sql.types.DataType

case class GpuScalaUDFLogical(udf: ScalaUDF) extends ShimExpression with Logging {
  override def nullable: Boolean = udf.nullable

  override def eval(input: InternalRow): Any = {
  }

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = {
    null
  }

  override def dataType: DataType = udf.dataType

  override def children: Seq[Expression] = udf.children

  def compile(isTestEnabled: Boolean): Expression = {
    compileWithDiagnostics(isTestEnabled) match {
      case Compiled(expression) => expression
      case Rejected(original, _, _) => original
    }
  }

  /**
   * Compiles the UDF and says what happened.
   *
   * The reason a rejection carries used to exist only as a debug log line, which meant nothing
   * downstream could act on it: a caller that wanted to try another route for a UDF this compiler
   * declined had no way to tell "a loop, which an expression tree cannot hold" from "a string
   * operation, which is not numeric code at all". Both answers are values now. The compilation
   * itself is unchanged, and so is what [[compile]] returns, so existing callers see exactly what
   * they saw before.
   */
  def compileWithDiagnostics(isTestEnabled: Boolean): UdfCompileResult = {
    val udfName = udf.udfName.getOrElse("<unknown>")
    try {
      val expr = CatalystExpressionBuilder(udf.function).compile(udf.children)
      if (expr.isDefined) {
        Compiled(expr.get)
      } else {
        Rejected(udf, Set(RejectionReason.Other),
          "the compiler produced no expression for the UDF")
      }
    } catch {
      case e: UdfRejectionException =>
        logDebug(s"UDF $udfName compilation failure (${e.reason.name}): $e")
        if (isTestEnabled) {
          throw e
        }
        Rejected(udf, Set(e.reason), Option(e.getMessage).getOrElse(""))
      case e: SparkException =>
        logDebug(s"UDF $udfName compilation failure: $e")
        if (isTestEnabled) {
          throw e
        }
        Rejected(udf, Set(RejectionReason.Other), Option(e.getMessage).getOrElse(""))
      case NonFatal(e) =>
        logWarning(s"Unable to translate UDF $udfName: $e")
        Rejected(udf, Set(RejectionReason.Other), Option(e.getMessage).getOrElse(""))
    }
  }
}
