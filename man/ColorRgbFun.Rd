% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/colorpatch_impl.R
\name{ColorRgbFun}
\alias{ColorRgbFun}
\title{Creates a color mapping function}
\usage{
ColorRgbFun(pal, xmin = -1, xmax = 1, coerce.fun = colorspace::hex)
}
\arguments{
\item{pal}{the color palette}

\item{xmin}{minimum value to be mapped to the first entry of the palette}

\item{xmax}{maximum value to be mapped to the last entry of the palette}

\item{coerce.fun}{the color coercing function (e.g. for ggplot2 \code{\link[colorspace:hex]{colorspace::hex()}} is recommended)}
}
\value{
a function mapping a value to a color
}
\description{
Creates a color mapping function
}
\examples{
data("OptimGreenRedLAB")
fn <- ColorRgbFun(OptimGreenRedLAB)
}
