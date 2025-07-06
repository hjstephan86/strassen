# graph_test.py

import unittest
from src.graph import Graph
from src.matrix import Matrix # To assert types and compare matrices

class TestGraph(unittest.TestCase):

    def test_init_valid_undirected(self):
        """Test Graph initialization for undirected graph."""
        g = Graph(4)
        self.assertEqual(g.num_nodes, 4)
        self.assertFalse(g.directed)
        expected_matrix = Matrix([[0, 0, 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0]])
        self.assertEqual(g.get_adjacency_matrix(), expected_matrix)

    def test_init_valid_directed(self):
        """Test Graph initialization for directed graph."""
        g = Graph(3, directed=True)
        self.assertEqual(g.num_nodes, 3)
        self.assertTrue(g.directed)
        expected_matrix = Matrix([[0, 0, 0],
                                  [0, 0, 0],
                                  [0, 0, 0]])
        self.assertEqual(g.get_adjacency_matrix(), expected_matrix)

    def test_init_invalid_num_nodes(self):
        """Test Graph initialization with invalid number of nodes."""
        with self.assertRaises(ValueError):
            Graph(0)
        with self.assertRaises(ValueError):
            Graph(-1)
        with self.assertRaises(ValueError):
            Graph(3.5)
        with self.assertRaises(ValueError):
            Graph("abc")

    def test_add_edge_undirected(self):
        """Test adding edges in an undirected graph."""
        g = Graph(3)
        g.add_edge(0, 1)
        self.assertTrue(g.has_edge(0, 1))
        self.assertTrue(g.has_edge(1, 0)) # Check symmetric
        self.assertFalse(g.has_edge(0, 2))
        
        g.add_edge(1, 2)
        expected_matrix = Matrix([[0, 1, 0],
                                  [1, 0, 1],
                                  [0, 1, 0]])
        self.assertEqual(g.get_adjacency_matrix(), expected_matrix)

        # Add existing edge, should not change anything
        g.add_edge(0, 1)
        self.assertEqual(g.get_adjacency_matrix(), expected_matrix)

        # Add self-loop
        g.add_edge(1, 1)
        expected_matrix_with_loop = Matrix([[0, 1, 0],
                                           [1, 1, 1], # [1][1] is now 1
                                           [0, 1, 0]])
        self.assertEqual(g.get_adjacency_matrix(), expected_matrix_with_loop)


    def test_add_edge_directed(self):
        """Test adding edges in a directed graph."""
        g = Graph(3, directed=True)
        g.add_edge(0, 1)
        self.assertTrue(g.has_edge(0, 1))
        self.assertFalse(g.has_edge(1, 0)) # Should not be symmetric
        
        g.add_edge(1, 2)
        expected_matrix = Matrix([[0, 1, 0],
                                  [0, 0, 1],
                                  [0, 0, 0]])
        self.assertEqual(g.get_adjacency_matrix(), expected_matrix)

        # Add existing edge
        g.add_edge(0, 1)
        self.assertEqual(g.get_adjacency_matrix(), expected_matrix)

        # Add self-loop
        g.add_edge(1, 1)
        expected_matrix_with_loop = Matrix([[0, 1, 0],
                                           [0, 1, 1], # [1][1] is now 1
                                           [0, 0, 0]])
        self.assertEqual(g.get_adjacency_matrix(), expected_matrix_with_loop)

    def test_add_edge_out_of_bounds(self):
        """Test adding edges with out-of-bounds nodes."""
        g = Graph(2)
        with self.assertRaises(IndexError):
            g.add_edge(0, 2)
        with self.assertRaises(IndexError):
            g.add_edge(2, 0)
        with self.assertRaises(IndexError):
            g.add_edge(-1, 0)

    def test_remove_edge_undirected(self):
        """Test removing edges in an undirected graph."""
        g = Graph(3)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(1, 1) # Add a self-loop

        g.remove_edge(0, 1)
        self.assertFalse(g.has_edge(0, 1))
        self.assertFalse(g.has_edge(1, 0)) # Should be symmetric
        self.assertTrue(g.has_edge(1, 2))
        self.assertTrue(g.has_edge(1, 1)) # Self-loop remains

        # Try to remove non-existent edge
        g.remove_edge(0, 2)
        expected_matrix = Matrix([[0, 0, 0],
                                  [0, 1, 1],
                                  [0, 1, 0]])
        self.assertEqual(g.get_adjacency_matrix(), expected_matrix)
        
        g.remove_edge(1, 1) # Remove self-loop
        expected_matrix_no_loop = Matrix([[0, 0, 0],
                                          [0, 0, 1],
                                          [0, 1, 0]])
        self.assertEqual(g.get_adjacency_matrix(), expected_matrix_no_loop)

    def test_remove_edge_directed(self):
        """Test removing edges in a directed graph."""
        g = Graph(3, directed=True)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(1, 1) # Add a self-loop

        g.remove_edge(0, 1)
        self.assertFalse(g.has_edge(0, 1))
        self.assertFalse(g.has_edge(1, 0)) # Still not symmetric
        self.assertTrue(g.has_edge(1, 2))
        self.assertTrue(g.has_edge(1, 1)) # Self-loop remains

        # Try to remove non-existent edge
        g.remove_edge(0, 2)
        expected_matrix = Matrix([[0, 0, 0],
                                  [0, 1, 1],
                                  [0, 0, 0]])
        self.assertEqual(g.get_adjacency_matrix(), expected_matrix)

        g.remove_edge(1, 1) # Remove self-loop
        expected_matrix_no_loop = Matrix([[0, 0, 0],
                                          [0, 0, 1],
                                          [0, 0, 0]])
        self.assertEqual(g.get_adjacency_matrix(), expected_matrix_no_loop)


    def test_remove_edge_out_of_bounds(self):
        """Test removing edges with out-of-bounds nodes."""
        g = Graph(2)
        with self.assertRaises(IndexError):
            g.remove_edge(0, 2)
        with self.assertRaises(IndexError):
            g.remove_edge(2, 0)
        with self.assertRaises(IndexError):
            g.remove_edge(-1, 0)

    def test_has_edge_basic(self):
        """Test has_edge for existing and non-existing edges."""
        g = Graph(3)
        g.add_edge(0, 1)
        self.assertTrue(g.has_edge(0, 1))
        self.assertTrue(g.has_edge(1, 0))
        self.assertFalse(g.has_edge(0, 2))
        self.assertFalse(g.has_edge(2, 1))

    def test_has_edge_out_of_bounds(self):
        """Test has_edge with out-of-bounds nodes."""
        g = Graph(2)
        with self.assertRaises(IndexError):
            g.has_edge(0, 2)
        with self.assertRaises(IndexError):
            g.has_edge(2, 0)
        with self.assertRaises(IndexError):
            g.has_edge(-1, 0)

    def test_get_adjacency_matrix(self):
        """Test get_adjacency_matrix returns correct Matrix object."""
        g = Graph(2)
        adj_matrix = g.get_adjacency_matrix()
        self.assertIsInstance(adj_matrix, Matrix)
        expected_matrix = Matrix([[0, 0], [0, 0]])
        self.assertEqual(adj_matrix, expected_matrix)
        
        g.add_edge(0, 1)
        expected_matrix_after_add = Matrix([[0, 1], [1, 0]])
        self.assertEqual(g.get_adjacency_matrix(), expected_matrix_after_add)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False) # Use exit=False to allow running other tests