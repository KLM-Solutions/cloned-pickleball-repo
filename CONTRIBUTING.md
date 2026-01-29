# Contributing to StrikeSense

Thank you for your interest in contributing to StrikeSense! This document provides guidelines for development and collaboration.

## 🚀 Getting Started

### 1. Setup Development Environment..

```bash
# Clone the repository
git clone <repository-url>
cd pickle-ball-main

# Install dependencies
npm install

# Run development server
npm run dev
```

### 2. Python Setup

```bash
cd python
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## 📝 Development Workflow

### Branch Naming Convention

- `feature/` - New features
- `fix/` - Bug fixes
- `refactor/` - Code refactoring
- `docs/` - Documentation updates

Example: `feature/add-new-stroke-type`

### Commit Messages

Follow conventional commits:

```
feat: add new stroke classification
fix: resolve video upload issue
docs: update README with deployment guide
refactor: optimize biomechanics calculations
```

## 🏗️ Code Structure

### Frontend (Next.js/React)

- **Components**: Reusable UI components in `app/components/`
- **Pages**: Route-based pages in `app/strikesense/`
- **Utilities**: Helper functions in `lib/`
- **Styles**: Tailwind CSS in `app/globals.css`

### Backend (Python)

- **Tracking**: `python/track.py`
- **Biomechanics**: `python/biomechanics/`
- **Models**: Auto-downloaded to `models/`

## 🎨 Code Style

### TypeScript/React

- Use TypeScript for type safety
- Follow React hooks best practices
- Use functional components
- Implement proper error handling

### Python

- Follow PEP 8 style guide
- Use type hints where applicable
- Document functions with docstrings
- Keep functions focused and small

## 🧪 Testing

### Before Submitting

1. Test all stroke types
2. Verify mobile responsiveness
3. Check video upload/processing flow
4. Ensure Python backend works correctly

### Manual Testing Checklist

- [ ] Home page loads correctly
- [ ] All 6 camera guides display properly
- [ ] Video upload works
- [ ] Player selection/cropping works
- [ ] Analysis completes successfully
- [ ] Results display correctly

## 📦 Building for Production

```bash
# Build Next.js app
npm run build

# Test production build
npm start
```

## 🐛 Reporting Issues

When reporting bugs, include:

1. **Description**: Clear description of the issue
2. **Steps to Reproduce**: Detailed steps
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Screenshots**: If applicable
6. **Environment**: OS, browser, Node.js version

## 💡 Feature Requests

For new features:

1. Describe the feature
2. Explain the use case
3. Provide mockups if applicable
4. Discuss implementation approach

## 📚 Resources

- [Next.js Documentation](https://nextjs.org/docs)
- [React Documentation](https://react.dev)
- [Tailwind CSS](https://tailwindcss.com/docs)
- [YOLOv8 Documentation](https://docs.ultralytics.com)

## 🤝 Code Review Process

1. Create a pull request
2. Ensure all checks pass
3. Request review from team members
4. Address feedback
5. Merge after approval

## 📞 Contact

For questions or discussions, contact the development team.

---

Happy coding! 🎾
