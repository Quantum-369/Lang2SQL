# Lang2SQL 📊

A modern, AI-powered natural language to SQL query translator built with Streamlit and Claude. Lang2SQL empowers business stakeholders, product managers, and intermediate coders to unlock data insights from traditional SQL databases without writing complex queries.

## Features 🚀

- **Natural Language Processing**: Convert plain English questions into optimized SQL queries
- **Interactive ERD Visualization**: Generate and view Entity Relationship Diagrams for selected database schemas
- **Quick Analysis**: Pre-generated analytical questions based on your database structure
- **Deep-Dive Analysis**: Custom analysis with the ability to build upon previous queries
- **Favorites System**: Save and manage frequently used queries
- **User Authentication**: Secure access with user authentication system
- **Self-Correcting Queries**: Automatic validation and correction of generated SQL queries

## Prerequisites 📋

- Python 3.7+
- MySQL Database
- Anthropic API Key (for Claude integration)

## Installation 🛠️

1. Clone the repository:
```bash
git clone https://github.com/yourusername/lang2sql.git
cd lang2sql
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env`:
```env
MYSQL_HOST=your_host
MYSQL_USER=your_user
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=your_database
ANTHROPIC_API_KEY=your_anthropic_api_key
```

4. Create authentication configuration in `authenticator.yml`:
```yaml
cookie:
  expiry_days: 30
  key: your_secret_key
  name: your_cookie_name
credentials:
  usernames:
    username1:
      email: user1@example.com
      name: User One
      password: hashed_password
preauthorized:
  emails:
    - user2@example.com
```

## Usage 🎯

1. Start the application:
```bash
streamlit run lang2sql_live.py
```

2. Login with your credentials

3. Select your database schema and tables from the sidebar

4. Use any of the following features:
   - Generate ERD diagram
   - Quick Analysis with pre-generated questions
   - Deep-dive Analysis with custom questions
   - Save favorite queries for later use

## Project Structure 📁

```
lang2sql/
├── lang2sql_live.py     # Main Streamlit application
├── utils.py             # Utility functions and database operations
├── authenticator.yml    # Authentication configuration
├── .env                 # Environment variables
└── requirements.txt     # Project dependencies
```

## Core Components 🔧

### Database Connection
- Configurable MySQL connection using environment variables
- Support for multiple schemas and tables
- Automatic schema analysis and relationship detection

### Query Generation
- Natural language processing using Claude AI
- SQL query validation and self-correction
- Support for complex joins and nested queries
- Query history tracking

### User Interface
- Modern, intuitive Streamlit interface
- Interactive ERD diagram generation
- Real-time query execution and results display
- Favorite query management system

## Security Features 🔒

- User authentication and session management
- Secure password handling
- Environment variable configuration
- SQL injection prevention
- Rate limiting and query size restrictions

## Contributing 🤝

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments 👏

- [Streamlit](https://streamlit.io/) for the amazing web framework
- [Anthropic](https://www.anthropic.com/) for Claude AI integration
- [MySQL](https://www.mysql.com/) for database support

## Support 💬

For support, please open an issue in the GitHub repository or contact the maintainers.

---
Built with ❤️ using Streamlit and Claude AI
