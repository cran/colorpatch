% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/colorpatch_impl.R
\name{ColorDistance}
\alias{ColorDistance}
\title{Computes the perceptional distance between two neighboring colors}
\usage{
ColorDistance(pal, color.space = "LAB")
}
\arguments{
\item{pal}{the color palette}

\item{color.space}{color space in which the distance shall be computed (default "LAB")}
}
\value{
a vector of distances
}
\description{
Computes the perceptional distance between two neighboring colors
}
\examples{
data("OptimGreenRedLAB")
dd <- ColorDistance(OptimGreenRedLAB)
}
