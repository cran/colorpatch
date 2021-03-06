% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/colorpatch_impl.R
\name{InterpolateColorFun}
\alias{InterpolateColorFun}
\title{Linear interpolation within a \link[colorspace:color-class]{colorspace::color} palette}
\usage{
InterpolateColorFun(pal, xmin = -1, xmax = +1,
  coerce.fun = colorspace::hex)
}
\arguments{
\item{pal}{The input palette (must be of class \link[colorspace:color-class]{colorspace::color})}

\item{xmin}{minimum of the numeric range to be mapped onto \code{pal}}

\item{xmax}{maximum of the numeric range to be mapped onto \code{pal}}

\item{coerce.fun}{each color will be coerced by this function (defaults to \code{\link[colorspace:hex]{colorspace::hex()}})}
}
\value{
A function mapping a numeric value \code{value} onto a color value.
}
\description{
This function can be used together with \link{ggplot2} for mapping values onto \link[colorspace:color-class]{colorspace::color} palettes.
The color is then coerced with \code{coerce.fun}.
}
\examples{
library(colorspace)
library(colorpatch)
data("OptimGreenRedLAB")
fn <- InterpolateColorFun(OptimGreenRedLAB)
cols <- fn(seq(-1, 1, by = 0.1))
specplot(cols)
}
