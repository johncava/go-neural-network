package main

import (
	"fmt"
	"image/color"
	"math"
	"math/rand"

	"github.com/gonum/matrix/mat64"
	"github.com/gonum/plot"
	"github.com/gonum/plot/plotter"
	"github.com/gonum/plot/vg"
	"github.com/gonum/plot/vg/draw"
)

//Applies the sigmoid function to each element of the matrix
func sigmoid(i, j int, v float64) float64 {
	s := 1 / (1 + math.Exp(-v))
	return s
}

//Applies the derivative of the sigmoid to each element of the matrix
func sigmoidDerivative(i, j int, v float64) float64 {
	s := v * (1 - v)
	return s
}

//Creates a matrix with random entries given number of ROWS and COLS
func randomData(rows int, cols int) *mat64.Dense {
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = rand.NormFloat64()
	}
	mat := mat64.NewDense(rows, cols, data)
	return mat
}

//Function that takes the absolute value of each error and takes the average for the matrix
func averageError(m *mat64.Dense) float64 {
	rows, cols := m.Dims()
	box := make([]float64, rows*cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			box = append(box, m.At(i, j))
		}
	}
	avg := 0.0
	for item := range box {
		avg = avg + math.Abs(box[item])
	}
	avg = avg / (float64(rows * cols))
	return avg
}

func createPoints(a []float64) plotter.XYs {
	pts := make(plotter.XYs, len(a))
	for i := range a {
		pts[i].X = float64(i)
		pts[i].Y = a[i]
		fmt.Print(i, " \n")
	}
	return pts
}

func main() {
	maxIterations := 100
	// Generate matrices of random values.
	input := randomData(4, 3)
	weights1 := randomData(3, 5)
	weights2 := randomData(5, 5)
	weights3 := randomData(5, 2)
	y := randomData(4, 2)

	linePointsData := make([]float64, 0)
	var plotingData plotter.XYs

	for iterations := 0; iterations < maxIterations; iterations++ {

		//Forward Propagation

		hiddenLayer1 := mat64.NewDense(4, 5, nil)
		hiddenLayer1.Mul(input, weights1)
		hiddenLayer1.Apply(sigmoid, hiddenLayer1)

		hiddenLayer2 := mat64.NewDense(4, 5, nil)
		hiddenLayer2.Mul(hiddenLayer1, weights2)
		hiddenLayer2.Apply(sigmoid, hiddenLayer2)

		output := mat64.NewDense(4, 2, nil)
		output.Mul(hiddenLayer2, weights3)
		output.Apply(sigmoid, output)

		outputError := mat64.NewDense(4, 2, nil)
		outputError.Sub(output, y)

		//Printing of the error per epoch
		if iterations%1 == 0 {
			//Computes the average error for the outputError for this epoch
			err := averageError(outputError)
			fmt.Print("Error: ", err)
			fmt.Print("\n")
			linePointsData = append(linePointsData, err)
		}

		//Back Propagation
		output.Apply(sigmoidDerivative, output)
		outputLayerDelta := mat64.NewDense(4, 2, nil)
		outputLayerDelta.MulElem(outputError, output)

		hiddenLayer2Error := mat64.NewDense(4, 5, nil)
		hiddenLayer2Error.Mul(outputLayerDelta, weights3.T())

		// fmt.Print("H2")

		hiddenLayer2Delta := mat64.NewDense(4, 5, nil)
		hl2 := mat64.NewDense(4, 5, nil)
		hl2.Apply(sigmoidDerivative, hiddenLayer2)
		hiddenLayer2Delta.MulElem(hiddenLayer2Error, hl2)

		// fmt.Print("H3")

		hiddenLayer1Error := mat64.NewDense(4, 5, nil)
		hiddenLayer1Error.Mul(hiddenLayer2Delta, weights2.T())

		// fmt.Print("H4")

		hiddenLayer1Delta := mat64.NewDense(4, 5, nil)
		hl1 := mat64.NewDense(4, 5, nil)
		hl1.Apply(sigmoidDerivative, hiddenLayer1)
		hiddenLayer1Delta.MulElem(hiddenLayer1Error, hl1)

		// fmt.Print("H5")

		//Gradient Descent
		hl2t := mat64.NewDense(5, 2, nil)
		hl2t.Mul(hiddenLayer2.T(), outputLayerDelta)
		weights3New := mat64.NewDense(5, 2, nil)
		weights3New.Sub(weights3, hl2t)
		weights3 = weights3New

		// fmt.Print("H6")

		hl1t := mat64.NewDense(5, 5, nil)
		hl1t.Mul(hiddenLayer1.T(), hiddenLayer2Delta)
		weights2New := mat64.NewDense(5, 5, nil)
		weights2New.Sub(weights2, hl1t)
		weights2 = weights2New

		// fmt.Print("H7")

		inputT := mat64.NewDense(3, 5, nil)
		inputT.Mul(input.T(), hiddenLayer1Delta)
		weights1New := mat64.NewDense(3, 5, nil)
		weights1New.Sub(weights1, inputT)
		weights1 = weights1New

		//fmt.Print("H8")
	}

	//Make Error Plot
	plotingData = createPoints(linePointsData)
	p, err := plot.New()
	if err != nil {
		panic(err)
	}
	p.Title.Text = "Neural Network Error"
	p.X.Label.Text = "Iterations"
	p.Y.Label.Text = "Error"

	// Make a line plotter with points and set its style.
	lpLine, lpPoints, err := plotter.NewLinePoints(plotingData)
	if err != nil {
		panic(err)
	}
	lpLine.Color = color.RGBA{G: 255, A: 255}
	lpPoints.Shape = draw.PyramidGlyph{}
	lpPoints.Color = color.RGBA{R: 255, A: 255}

	// Add the plotters to the plot, with a legend
	// entry for each
	p.Add(lpLine, lpPoints)
	p.Legend.Add("line points", lpLine, lpPoints)

	// Save the plot to a PNG file.
	if err := p.Save(4*vg.Inch, 4*vg.Inch, "error.png"); err != nil {
		panic(err)
	}
}
