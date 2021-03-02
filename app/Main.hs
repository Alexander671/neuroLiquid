{-# LANGUAGE BangPatterns, 
             ScopedTypeVariables,
             RecordWildCards,
             FlexibleContexts,
             TypeFamilies,
             DeriveGeneric #-}
module Main where

import Data.IDX ( decodeIDXFile, idxIntContent, IDXData ) 
{------------------------}
import Numeric.LinearAlgebra as NL ( Vector, fromList )
{------------------------}
import Data.Vector.Unboxed as DTU ( toList )

import Prelude hiding (readFile)
import Neuro
    ( Samples, Sample, createNetwork, tanh', saveNetwork, trainNTimes )


main :: IO ()
main = do
  n <- createNetwork 784 [64] 10
  samples <- importTrain
  {-----------------------------}
  let n' = trainNTimes 50 5.0 tanh tanh' n samples
  saveNetwork "smartNet5.nn" n'


importTrain :: IO (Samples Double)
importTrain = do
    Just idxTrain  <- decodeIDXFile "train-images.idx3-ubyte" -- image
    Just idxResult <- decodeIDXFile "train-labels.idx1-ubyte" -- result
    return $ samples idxTrain idxResult

samples :: IDXData -> IDXData -> [Sample Double]
samples idxTrain idxResult = Prelude.zip (image idxTrain) (result idxResult) :: [Sample Double]

image  = matrix2x2 . _0_1to0_255 . DTU.toList . idxIntContent
result = unitar . DTU.toList . idxIntContent

unitar :: [Int] -> [NL.Vector Double]
unitar []     = [] 
unitar (0:xs) = NL.fromList [1,-1,-1,-1,-1,-1,-1,-1,-1,-1] : unitar xs
unitar (1:xs) = NL.fromList [-1,1,-1,-1,-1,-1,-1,-1,-1,-1] : unitar xs
unitar (2:xs) = NL.fromList [-1,-1,1,-1,-1,-1,-1,-1,-1,-1] : unitar xs
unitar (3:xs) = NL.fromList [-1,-1,-1,1,-1,-1,-1,-1,-1,-1] : unitar xs
unitar (4:xs) = NL.fromList [-1,-1,-1,-1,1,-1,-1,-1,-1,-1] : unitar xs
unitar (5:xs) = NL.fromList [-1,-1,-1,-1,-1,1,-1,-1,-1,-1] : unitar xs
unitar (6:xs) = NL.fromList [-1,-1,-1,-1,-1,-1,1,-1,-1,-1] : unitar xs
unitar (7:xs) = NL.fromList [-1,-1,-1,-1,-1,-1,-1,1,-1,-1] : unitar xs
unitar (8:xs) = NL.fromList [-1,-1,-1,-1,-1,-1,-1,-1,1,-1] : unitar xs
unitar (9:xs) = NL.fromList [-1,-1,-1,-1,-1,-1,-1,-1,-1,1] : unitar xs

_0_1to0_255 :: [Int] -> [Double]
_0_1to0_255 [] = []
_0_1to0_255 (x:xs) = (fromIntegral x / 255) : _0_1to0_255 xs

matrix2x2 :: [Double] -> [NL.Vector Double]
matrix2x2 [] = []
matrix2x2 xs = NL.fromList (Prelude.take 784 xs) : matrix2x2 (Prelude.drop 784 xs)

