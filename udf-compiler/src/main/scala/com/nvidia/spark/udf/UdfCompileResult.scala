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

package com.nvidia.spark.udf

import org.apache.spark.SparkException
import org.apache.spark.sql.catalyst.expressions.{Expression, ScalaUDF}

/**
 * Why this compiler declined a UDF.
 *
 * The compiler used to answer only "yes, here is a Catalyst expression" or "no", logging the
 * reason at debug level and dropping it. A caller that wants to try something else with the UDF
 * needs the reason, so it is a value now rather than a log line. The categories are coarse on
 * purpose: they are cheap triage for whoever asks next, not a description of the UDF.
 */
sealed abstract class RejectionReason(val name: String)

object RejectionReason {
  /** A loop. Expression trees have no way to express one. */
  case object Loop extends RejectionReason("LOOP")

  /** An `iinc`, which in practice means a counted loop or an accumulating index. */
  case object Iinc extends RejectionReason("IINC")

  /** A primitive array load or store. */
  case object PrimitiveArray extends RejectionReason("PRIMITIVE_ARRAY")

  /** A maths function with no Catalyst equivalent. */
  case object UnsupportedMath extends RejectionReason("UNSUPPORTED_MATH")

  /** A call this compiler cannot inline or map. */
  case object UnsupportedInvoke extends RejectionReason("UNSUPPORTED_INVOKE")

  /** A field read or write on an object. */
  case object ObjectAccess extends RejectionReason("OBJECT_ACCESS")

  /** `athrow`, a try/catch, or anything else with exception semantics. */
  case object ExceptionHandling extends RejectionReason("EXCEPTION_HANDLING")

  /** The user's UDF could not be found behind Spark's wrapper. */
  case object WrapperExtraction extends RejectionReason("WRAPPER_EXTRACTION")

  /** A bytecode this compiler does not implement, outside the categories above. */
  case object UnsupportedBytecode extends RejectionReason("UNSUPPORTED_BYTECODE")

  /** Anything else, including a failure that carried no classification. */
  case object Other extends RejectionReason("OTHER")
}

/**
 * A [[SparkException]] that says which kind of construct stopped the compilation.
 *
 * It is thrown at the site that found the construct, so the reason is what that site knows rather
 * than something inferred later from the text of a message. Sites that have not been given a
 * reason yet keep throwing a plain `SparkException` and are reported as
 * [[RejectionReason.Other]], with their message carried through as diagnostics.
 */
class UdfRejectionException(val reason: RejectionReason, message: String)
    extends SparkException(message)

/** The outcome of compiling one UDF. */
sealed trait UdfCompileResult

/** The UDF became this Catalyst expression. */
case class Compiled(expression: Expression) extends UdfCompileResult

/**
 * The UDF stays as it is. `reasons` is what the compiler ran into -- in practice one reason, since
 * it stops at the first -- and `diagnostics` is the message that came with it.
 */
case class Rejected(
    original: ScalaUDF,
    reasons: Set[RejectionReason],
    diagnostics: String) extends UdfCompileResult {

  def reasonNames: Array[String] = reasons.map(_.name).toArray
}
