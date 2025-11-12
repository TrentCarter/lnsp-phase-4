import React, { useState, useEffect } from 'react';
import axios from 'axios';

function App() {
  const [directoryContents, setDirectoryContents] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [fileContent, setFileContent] = useState('');

  useEffect(() => {
    fetchDirectoryContents();
  }, []);

  async function fetchDirectoryContents(directory = "/Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4") {
    try {
      const response = await axios.get(`http://localhost:6102/api/files/list?directory=${encodeURIComponent(directory)}`);
      setDirectoryContents(response.data);
    } catch (error) {
      console.error('Error fetching directory contents:', error);
    }
  }

  async function searchFiles(query) {
    try {
      const response = await axios.get(`http://localhost:6102/api/files/search?query=${encodeURIComponent(query)}`);
      setDirectoryContents(response.data);
    } catch (error) {
      console.error('Error searching files:', error);
    }
  }

  async function getFileContent(path) {
    try {
      const response = await axios.get(`http://localhost:6102/api/files/content?path=${encodeURIComponent(path)}`);
      setFileContent(response.data.content);
    } catch (error) {
      console.error('Error getting file content:', error);
    }
  }

  async function editFileContent(path, content) {
    try {
      await axios.put(`http://localhost:6102/api/files/edit?path=${encodeURIComponent(path)}`, { content });
      fetchDirectoryContents();
    } catch (error) {
      console.error('Error editing file content:', error);
    }
  }

  return (
    <div className="App">
      <h1>File Manager</h1>
      <input type="text" placeholder="Search files..." value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} />
      <button onClick={() => searchFiles(searchQuery)}>Search</button>
      <ul>
        {directoryContents.map(item => (
          <li key={item.path}>
            {item.is_dir ? <span>{item.name}/</span> : <span>{item.name}</span>}
            {item.is_dir && <button onClick={() => fetchDirectoryContents(item.path)}>Open</button>}
            {!item.is_dir && <button onClick={() => getFileContent(item.path)}>View</button>}
          </li>
        ))}
      </ul>
      {fileContent && (
        <div>
          <h2>File Content</h2>
          <textarea value={fileContent} onChange={(e) => setFileContent(e.target.value)} />
          <button onClick={() => editFileContent(fileContent.path, fileContent)}>Save</button>
        </div>
      )}
    </div>
  );
}

export default App;
