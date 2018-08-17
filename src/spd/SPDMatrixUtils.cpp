/*
 *  SPDMatrixUtils.cpp
 *  SPDLIB
 *
 *  Created by Pete Bunting on 31/01/2011.
 *  Copyright 2011 SPDLib. All rights reserved.
 *
 *  This file is part of SPDLib.
 *
 *  SPDLib is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  SPDLib is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with SPDLib.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "spd/SPDMatrixUtils.h"


namespace spdlib{
	
	
	SPDMatrixUtils::SPDMatrixUtils()
	{
		
	}
	
	Matrix* SPDMatrixUtils::createMatrix(int n, int m) throw(SPDException)
	{
		/// Create matrix n rows by m colums
		if(n < 1 | m < 1)
		{
			throw SPDException("Sizes of m and n must be at least 1.");
		}
		Matrix *matrix = new Matrix();
		matrix->n = n;
		matrix->m = m;
		matrix->matrix = new double[n*m];
		
		int length = n * m;
		
		for(int i = 0; i < length; i++)
		{
			matrix->matrix[i] = 0;
		}
		return matrix;
	}
	
	Matrix* SPDMatrixUtils::createMatrix(Matrix *matrix) throw(SPDException)
	{
		if(matrix == NULL)
		{
			throw SPDException("The matrix to be copied is NULL.");
		}
		
		Matrix *newMatrix = new Matrix();
		newMatrix->n = matrix->n;
		newMatrix->m = matrix->m;
		newMatrix->matrix = new double[matrix->n*matrix->m];
		
		int length = newMatrix->n * newMatrix->m;
		
		for(int i = 0; i < length; i++)
		{
			newMatrix->matrix[i] = 0;
		}
		
		return newMatrix;
	}
	
	Matrix* SPDMatrixUtils::copyMatrix(Matrix *matrix) throw(SPDException)
	{
		if(matrix == NULL)
		{
			throw SPDException("The matrix to be copied is NULL.");
		}
		
		Matrix *newMatrix = new Matrix();
		newMatrix->n = matrix->n;
		newMatrix->m = matrix->m;
		newMatrix->matrix = new double[matrix->n*matrix->m];
		
		int numPTS = matrix->m * matrix->n;
		
		for(int i = 0; i < numPTS; i++)
		{
			newMatrix->matrix[i] = matrix->matrix[i];
		}
		return newMatrix;
	}
	
	void SPDMatrixUtils::freeMatrix(Matrix *matrix)
	{
		if(matrix != NULL)
		{
			if(matrix->matrix != NULL)
			{
				delete matrix->matrix;
			}
			delete matrix;
		}
	}
	
	double SPDMatrixUtils::determinant(Matrix *matrix) throw(SPDException)
	{
		double sum = 0;
		if(matrix->n != matrix->m)
		{
			throw SPDException("To calculate a determinant the matrix needs to be symatical");
		}
		
		if(matrix->n == 2)
		{
			sum = (matrix->matrix[0] * matrix->matrix[3]) - (matrix->matrix[1] * matrix->matrix[2]);
		}
		else
		{
			Matrix *tmpMatrix = NULL;
			double value = 0;
			int index = 0;
			for(int i = 0; i < matrix->n; i++)
			{
				
				index = 0;
				tmpMatrix = this->createMatrix((matrix->n-1), (matrix->m-1));
				
				// Populate new matrix
				for(int n = 1; n < matrix->n; n++)
				{
					for(int m = 0; m < matrix->m; m++)
					{
						if(i == m)
						{
							continue;
						}
						
						tmpMatrix->matrix[index] = matrix->matrix[(n*matrix->n)+m];
						index++;
					}
				}
				// Calculate the determinant of the new matrix
				value = this->determinant(tmpMatrix);
				
				// add to sum
				sum = sum + (pow(-1.0,i) * value * matrix->matrix[i]);
				
				this->freeMatrix(tmpMatrix);
			}
		}
		return sum;
	}
	
	Matrix* SPDMatrixUtils::cofactors(Matrix *matrix) throw(SPDException)
	{
		if(matrix->n != matrix->m)
		{
			throw SPDException("To calculate cofactors the matrix needs to be symatical");
		}
		Matrix *newMatrix = NULL;
		newMatrix = this->createMatrix(matrix->n, matrix->m);
		
		int index = 0;
		Matrix *tmpMatrix = this->createMatrix((matrix->n-1), (matrix->m-1));
		for(int i = 0; i < matrix->n; i++)
		{
			for(int j = 0; j < matrix->m; j++)
			{
				index = 0;
				for(int n = 0; n < matrix->n; n++)
				{
					if(i == n)
					{
						continue;
					}
					for(int m = 0; m < matrix->m; m++)
					{
						if(j == m)
						{
							continue;
						}
						tmpMatrix->matrix[index] = matrix->matrix[(n*matrix->n)+m];
						index++;
					}
				}
				newMatrix->matrix[(i*matrix->n)+j] = (pow(-1.0,((i*matrix->n)+j))) * this->determinant(tmpMatrix);
			}
		}
		this->freeMatrix(tmpMatrix);
		return newMatrix;
	}
	
	Matrix* SPDMatrixUtils::transpose(Matrix *matrix) throw(SPDException)
	{
		Matrix *newMatrix = NULL;
		newMatrix = this->createMatrix(matrix->m, matrix->n);
		for(int i = 0; i < matrix->n; i++)
		{
			for(int j = 0; j < matrix->m; j++)
			{
				newMatrix->matrix[i+(j*matrix->n)] = matrix->matrix[(i*matrix->n)+j];
			}
		}
		return newMatrix;
	}
	
	void SPDMatrixUtils::multipleSingle(Matrix *matrix, double multiple) throw(SPDException)
	{
		int numElements = matrix->n * matrix->m;
		for(int i = 0; i < numElements; i++)
		{
			matrix->matrix[i] = matrix->matrix[i] * multiple;
		}
	}
	
	Matrix* SPDMatrixUtils::multiplication(Matrix *matrixA, Matrix *matrixB) throw(SPDException)
	{
		Matrix *matrix1 = NULL;
		Matrix *matrix2 = NULL;
		if(matrixA->n == matrixB->m)
		{
			matrix1 = matrixA;
			matrix2 = matrixB;
		}
		else if(matrixA->m == matrixB->n)
		{
			matrix1 = matrixB;
			matrix2 = matrixA;
		}
		else
		{
			throw SPDException("Multipication required the number of columns to match the number of rows.");
		}
		//std::cout << "Creating new matrix\n";
		//std::cout << "matrix2->n = " << matrix2->n << std::endl;
		//std::cout << "matrix1->m = " << matrix1->m << std::endl;
		Matrix *newMatrix = this->createMatrix(matrix2->n, matrix1->m);
		
		double value = 0;
		int row = 0;
		int col = 0;
		int index = 0;
		for(int i = 0; i < matrix1->m; i++)
		{
			for(int j = 0; j < matrix2->n; j++)
			{
				value = 0;
				for(int n = 0; n < matrix1->n; n++)
				{
					row = (i * matrix1->n) + n;
					col = (n * matrix2->n) + j;
					//std::cout << "row = " << row << " col = " << col << std::endl;
					value += matrix1->matrix[row] * matrix2->matrix[col];
				}
				newMatrix->matrix[index] = value;
				index++;
			}
		}
		
		return newMatrix;
	}
	
	void SPDMatrixUtils::printMatrix(Matrix *matrix)
	{		
		int index = 0;
		for(int i = 0; i < matrix->n; i++)
		{
			for(int j = 0; j < matrix->m; j++)
			{
				std::cout << matrix->matrix[index++] << " ";
			}
			std::cout << std::endl;
		}
	}
	
	void SPDMatrixUtils::saveMatrix2GridTxt(Matrix *matrix, std::string filepath) throw(SPDException)
	{
		std::string outputFilename = filepath + std::string(".gmtxt");
		std::ofstream outTxtFile;
		outTxtFile.open(outputFilename.c_str(), std::ios::out | std::ios::trunc);
		
		if(outTxtFile.is_open())
		{
			outTxtFile << "m=" << matrix->m << std::endl;
			outTxtFile << "n=" << matrix->n << std::endl;
			
			int totalElements = matrix->n * matrix->m;
			int lastElement = totalElements-1;
			for(int i = 0; i < totalElements; i++)
			{
				if(i %  matrix->m == 0)
				{
					outTxtFile << std::endl;
				}
				if(i == lastElement)
				{
					outTxtFile << matrix->matrix[i];
				}
				else
				{
					outTxtFile << matrix->matrix[i] << ",";
				}
			}
			outTxtFile.flush();
			outTxtFile.close();
		}
		else
		{
			throw SPDException("Could not open text file.");
		}
		
	}
	
	void SPDMatrixUtils::saveMatrix2CSV(Matrix *matrix, std::string filepath) throw(SPDException)
	{
		std::string outputFilename = filepath + std::string(".csv");
		std::ofstream outTxtFile;
		outTxtFile.open(outputFilename.c_str(), std::ios::out | std::ios::trunc);
		
		if(outTxtFile.is_open())
		{			
			int totalElements = matrix->n * matrix->m;
			int lastElement = totalElements-1;
			for(int i = 0; i < totalElements; i++)
			{
				if(i %  matrix->m == 0)
				{
					outTxtFile << std::endl;
				}
				if(i == lastElement)
				{
					outTxtFile << matrix->matrix[i];
				}
				else
				{
					outTxtFile << matrix->matrix[i] << ",";
				}
			}
			outTxtFile.flush();
			outTxtFile.close();
		}
		else
		{
			throw SPDException("Could not open text file.");
		}
		
	}
	
	void SPDMatrixUtils::saveMatrix2txt(Matrix *matrix, std::string filepath) throw(SPDException)
	{
		std::string outputFilename = filepath + std::string(".mtxt");
		std::ofstream outTxtFile;
		outTxtFile.open(outputFilename.c_str(), std::ios::out | std::ios::trunc);
		
		if(outTxtFile.is_open())
		{
			outTxtFile << "m=" << matrix->m << std::endl;
			outTxtFile << "n=" << matrix->n << std::endl;
			
			int totalElements = matrix->n * matrix->m;
			int lastElement = totalElements-1;
			for(int i = 0; i < totalElements; i++)
			{
				if(i == lastElement)
				{
					outTxtFile << matrix->matrix[i];
				}
				else
				{
					outTxtFile << matrix->matrix[i] << ",";
				}
			}
			outTxtFile.flush();
			outTxtFile.close();
		}
		else
		{
			throw SPDException("Could not open text file.");
		}
	}
	
	void SPDMatrixUtils::saveMatrix2Binary(Matrix *matrix, std::string filepath) throw(SPDException)
	{
		std::ofstream matrixOutput;
		std::string matrixFilepath = filepath + std::string(".mtx");
		matrixOutput.open(matrixFilepath.c_str(), std::ios::out | std::ios::trunc | std::ios::binary);
		if(!matrixOutput.is_open())
		{
			throw SPDException("Could not open output stream for Matrix output.");
		}
		
		matrixOutput.write((char *) matrix->m, 4);
		matrixOutput.write((char *) matrix->n, 4);
		
		int matrixLength = matrix->m * matrix->n;
		for(int i = 0; i < matrixLength; i++)
		{
			matrixOutput.write((char *) &matrix->matrix[i], 4);
		}
		
		matrixOutput.flush();
		matrixOutput.close();
	}
	
	Matrix* SPDMatrixUtils::readMatrixFromTxt(std::string filepath) throw(SPDException)
	{
		SPDTextFileUtilities textUtils;
		Matrix *matrix = new Matrix();
		std::ifstream inputMatrix;
		inputMatrix.open(filepath.c_str());
		if(!inputMatrix.is_open())
		{
			throw SPDException("Could not open input text file.");
		}
		else
		{
			std::string strLine;
			std::string word;
			int number;
			float value;
			int lineCounter = 0;
			inputMatrix.seekg(std::ios_base::beg);
			while(!inputMatrix.eof())
			{
				getline(inputMatrix, strLine, '\n');
				if(strLine.length() > 0)
				{
					if(lineCounter == 0)
					{
						// m
						word = strLine.substr(2);
						number = textUtils.strto32bitUInt(word);
						matrix->m = number;
					}
					else if(lineCounter == 1)
					{
						// n
						word = strLine.substr(2);
						number = textUtils.strto32bitUInt(word);
						matrix->n = number;
					}
					else if(lineCounter == 2)
					{
						// data
						int dataCounter = 0;
						int start = 0;
						int lineLength = strLine.length();
						int numDataPoints = matrix->n*matrix->m;
						matrix->matrix = new double[numDataPoints];
						for(int i = 0; i < lineLength; i++)
						{
							if(strLine.at(i) == ',')
							{
								word = strLine.substr(start, i-start);								
								value = textUtils.strtodouble(word);
								if(boost::math::isnan(value))
								{
									value = 0;
								}
								matrix->matrix[dataCounter] = value;
								dataCounter++;
								
								start = start + i-start+1;
							}
							
							if(dataCounter >= numDataPoints)
							{
								throw SPDException("Too many data values, compared to header.");
							}
						}
						word = strLine.substr(start);
						value = textUtils.strtodouble(word);
						matrix->matrix[dataCounter] = value;
						dataCounter++;
						
						if(dataCounter != (matrix->n*matrix->m))
						{
							throw SPDException("An incorrect number of data points were read in.");
						}
						
					}
					else
					{
						break;
					}
				}
				lineCounter++;
			}
			
			if(lineCounter < 3)
			{
				throw SPDException("A complete matrix has not been reconstructed.");
			}
			inputMatrix.close();
		}
		return matrix;
	}
	
	Matrix* SPDMatrixUtils::readMatrixFromGridTxt(std::string filepath) throw(SPDException)
	{
        SPDTextFileUtilities txtUtils;
		Matrix *matrix = new Matrix();
		std::ifstream inputMatrix;
		inputMatrix.open(filepath.c_str());
		if(!inputMatrix.is_open())
		{
			throw SPDException("Could not open input text file.");
		}
		else
		{
			std::string strLine;
			std::string wholeline;
			std::string word;
			int number;
			float value;
			int lineCounter = 0;
			bool first = true;
			inputMatrix.seekg(std::ios_base::beg);
			while(!inputMatrix.eof())
			{
				getline(inputMatrix, strLine, '\n');
				if(strLine.length() > 0)
				{
					if(lineCounter == 0)
					{
						// m
						word = strLine.substr(2);
						number = txtUtils.strto32bitInt(word);
						matrix->m = number;
						//std::cout << "columns = " << number << std::endl;
					}
					else if(lineCounter == 1)
					{
						// n
						word = strLine.substr(2);
						number = txtUtils.strto32bitInt(word);
						matrix->n = number;
						//std::cout << "rows = " << number << std::endl;
					}
					else
					{
						if(first)
						{
							wholeline = strLine;
							first = false;
						}
						else
						{
							wholeline = wholeline + std::string(",") + strLine;
						}
					}
					lineCounter++;
				}
			}
			inputMatrix.close();
			
			// data
			int dataCounter = 0;
			int start = 0;
			int lineLength = wholeline.length(); ;
			int numDataPoints = matrix->n*matrix->m;
			matrix->matrix = new double[numDataPoints];
			
			for(int i = 0; i < lineLength; i++)
			{
				if(wholeline.at(i) == ',')
				{
					word = wholeline.substr(start, i-start);								
					value = txtUtils.strtodouble(word);
					matrix->matrix[dataCounter] = value;
					dataCounter++;
					
					start = start + i-start+1;
				}
				
				if(dataCounter >= numDataPoints)
				{
					throw SPDException("Too many data values, compared to header.");
				}
			}
			
			word = wholeline.substr(start);
			value = txtUtils.strtodouble(word);
			matrix->matrix[dataCounter] = value;
			dataCounter++;
			
			if(dataCounter != (matrix->n*matrix->m))
			{
				throw SPDException("An incorrect number of data points were read in.");
			}
		}
		return matrix;
	}
	
	
	Matrix* SPDMatrixUtils::readMatrixFromBinary(std::string filepath) throw(SPDException)
	{
		Matrix *matrix = new Matrix();
		std::string matrixFilepath = filepath + std::string(".mtx");
		std::ifstream matrixInput;
		matrixInput.open(matrixFilepath.c_str(), std::ios::in | std::ios::binary);
		if(!matrixInput.is_open())
		{
			throw SPDException("Could not open matrix binary file.");
		}
		
		matrixInput.seekg (0, std::ios::end);
		long end = matrixInput.tellg();
		matrixInput.seekg (0, std::ios::beg);
		int matrixSizeFile = (end/16) - 2;
		
		matrixInput.read((char *) &matrix->m, 4);
		matrixInput.read((char *) &matrix->n, 4);
		
		int matrixSize = matrix->m * matrix->n;
		if(matrixSizeFile != matrixSize)
		{
			throw SPDException("The file size and header differ on the number of points.");
		}
		
		matrix->matrix = new double[matrixSize];
		
		for(int i = 0; i < matrixSize; i++)
		{
			matrixInput.read((char *) &matrix->matrix[i], 4);
		}
		
		matrixInput.close();
		return matrix;
	}
	
	Matrix* SPDMatrixUtils::normalisedMatrix(Matrix *matrix, double min, double max) throw(SPDException)
	{
		double matrixMIN = 0;
		double matrixMAX = 0;
		double matrixDIFF = 0;
		double inDIFF = 0;
		bool first = true;
		int size = matrix->m * matrix->n;
		
		for(int i = 0; i < size; i++)
		{
			if(first)
			{
				matrixMIN = matrix->matrix[i];
				matrixMAX = matrix->matrix[i];
				first = false;
			}
			else
			{
				if( matrix->matrix[i] > matrixMAX)
				{
					matrixMAX = matrix->matrix[i];
				}
				else if( matrix->matrix[i] < matrixMIN)
				{
					matrixMIN = matrix->matrix[i];
				}
			}
		}
		
		inDIFF = max - min;
		matrixDIFF = matrixMAX - matrixMIN;
		
		Matrix *outMatrix = this->createMatrix(matrix->n, matrix->m);
		double norm = 0;
		
		for(int i = 0; i < size; i++)
		{
			norm = (matrix->matrix[i] - matrixMIN)/matrixDIFF;
			outMatrix->matrix[i] = (norm * inDIFF) + min;
		}
		
		return outMatrix;
	}
	
	Matrix* SPDMatrixUtils::duplicateMatrix(Matrix *matrix, int xDuplications, int yDuplications) throw(SPDException)
	{
		int newM = matrix->m * xDuplications;
		int newN = matrix->n * yDuplications;
		
		Matrix *outMatrix = this->createMatrix(newN, newM);
		
		int column = 0;
		int row = 0;
		int width = matrix->m;
		int height = matrix->n;
		int length = (width * xDuplications) * height;
		
		int inCounter = 0;
		int outCounter = 0;
		int xDupCount = 0;
		
		for(int n = 0; n < yDuplications; n++)
		{
			inCounter = 0;
			xDupCount = 0;
			outCounter = n * length;
			
			for(int i = 0; i < length; i++)
			{
				//std::cout << "[" << column << "," << row << "]: " << xDupCount <<  " out = " << outCounter << " in = " << inCounter << std::endl;
				
				outMatrix->matrix[outCounter++] = matrix->matrix[inCounter++];
				
				column++;
				if(column == width)
				{
					xDupCount++;
					
					if(xDupCount < xDuplications)
					{
						inCounter = inCounter - width;
					}
					else
					{
						row++;
						xDupCount = 0;
					}
					column = 0;
				}
			}
		}		
		
		return outMatrix;
	}
	
	SPDMatrixUtils::~SPDMatrixUtils()
	{
		
	}
}

