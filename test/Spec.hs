{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE BangPatterns, 
             ScopedTypeVariables,
             RecordWildCards,
             FlexibleContexts,
             TypeFamilies #-}
import Data.IDX (IDXData,  decodeIDXFile, idxIntContent ) 
{------------------------}
import Numeric.LinearAlgebra as NL (cmap, (#>), vjoin, Numeric,  Vector, fromList )
{------------------------}
import Data.Vector.Unboxed as DTU ( toList )
import qualified Data.Vector.Storable as DT
import Data.List (sort)
import Prelude hiding (readFile)
import Neuro
    ( ActivationFunction, Network(..), loadNetwork )


kol_vo :: Int
kol_vo = 1000

main :: IO ()
main = do
        res <- loadNetwork "smartNet5.nn" 
        Just idxTrain  <- decodeIDXFile "test/t10k-images.idx3-ubyte" -- image
        Just idxResult <- decodeIDXFile "test/t10k-labels.idx1-ubyte" -- result
        --writeFile "resNeuro.txt" (concat $ take (28*kol_vo) $ drowImage idxTrain)
        --writeFile "resLabel.txt" (show $ take kol_vo $ DTU.toList $  idxIntContent idxResult)
        putStrLn "Точность алгоритма :"
        print $ (\x -> (fromIntegral x) / fromIntegral kol_vo) $ length $ filter (\((_,x),y) -> x == y) $ flip zip (filterCorrect idxResult) $  printRes kol_vo res idxTrain $ DTU.toList $ idxIntContent idxResult 
        where filterCorrect idxResult = reverse $ take kol_vo $ DTU.toList $ idxIntContent idxResult

printRes :: Int -> Network Double -> IDXData -> t -> [(Double,Int)]
printRes 0 _   _           _      = []
printRes x res idxTrain idxResult = last (sort $ flip zip [0..] $ softmax $ DT.toList $ Main.output res tanh (last $ take x $ image idxTrain)) : printRes (x-1) res idxTrain idxResult

output :: (Floating (Vector a), Numeric a, DT.Storable a, Num (Vector a)) => Network a -> ActivationFunction a -> Vector a -> Vector a
output (Network{..}) act input = foldl f (vjoin [input, 1]) matrices
  where f (!inp) m = cmap act $ m #> inp

softmax :: [Double] -> [Double]
softmax xs = let xs' = exp <$> xs
                 s   = sum xs'
             in map (/ s) xs'

drowImage :: IDXData -> [String]
drowImage = _0_255to0or1 . DTU.toList . idxIntContent

_0_255to0or1 :: [Int] -> [String]
_0_255to0or1 [] = []
_0_255to0or1 xs = (show (concat $ fmap show $ convert $ take 28 xs) ++ " \n") : _0_255to0or1 (drop 28 xs)

convert [] = []
convert (x:xs) 
              | x > 240 = 1 : convert xs
              | otherwise = 0 : convert xs


image :: IDXData -> [Vector Double]
image  = matrix2x2 . _0_1to0_255 . DTU.toList . idxIntContent
result :: IDXData -> [Vector Double]
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
