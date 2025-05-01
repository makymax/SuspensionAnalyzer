import os
import sqlalchemy
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import datetime
import json

# Create SQLAlchemy base class
Base = declarative_base()

# Define database models
class AnalysisSession(Base):
    """Model for storing motorcycle suspension analysis sessions"""
    __tablename__ = 'analysis_sessions'
    
    id = Column(Integer, primary_key=True)
    session_name = Column(String(100), nullable=False)
    motorcycle_info = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    video_filename = Column(String(255), nullable=True)
    video_duration = Column(Float, nullable=True)
    video_fps = Column(Float, nullable=True)
    
    # Relationship to metrics
    metrics = relationship("AnalysisMetrics", back_populates="session", uselist=False, cascade="all, delete-orphan")
    
    # Relationship to suspension data points
    data_points = relationship("SuspensionDataPoint", back_populates="session", cascade="all, delete-orphan")
    
    # Relationship to recommended settings
    recommendations = relationship("SuspensionRecommendation", back_populates="session", cascade="all, delete-orphan")

class AnalysisMetrics(Base):
    """Model for storing calculated suspension metrics"""
    __tablename__ = 'analysis_metrics'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('analysis_sessions.id'), nullable=False)
    avg_travel = Column(Float, nullable=True)
    max_travel = Column(Float, nullable=True)
    min_travel = Column(Float, nullable=True)
    travel_range = Column(Float, nullable=True)
    max_compression_time = Column(Float, nullable=True)
    avg_compression_speed = Column(Float, nullable=True)
    avg_rebound_speed = Column(Float, nullable=True)
    max_compression_speed = Column(Float, nullable=True)
    max_rebound_speed = Column(Float, nullable=True)
    comp_rebound_ratio = Column(Float, nullable=True)
    oscillation_frequency = Column(Float, nullable=True)
    damping_ratio = Column(Float, nullable=True)
    
    # Relationship back to session
    session = relationship("AnalysisSession", back_populates="metrics")

class SuspensionDataPoint(Base):
    """Model for storing individual suspension data points"""
    __tablename__ = 'suspension_data_points'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('analysis_sessions.id'), nullable=False)
    time = Column(Float, nullable=False)
    distance = Column(Float, nullable=False)
    velocity = Column(Float, nullable=True)
    
    # Relationship back to session
    session = relationship("AnalysisSession", back_populates="data_points")

class SuspensionRecommendation(Base):
    """Model for storing suspension setting recommendations"""
    __tablename__ = 'suspension_recommendations'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('analysis_sessions.id'), nullable=False)
    component = Column(String(100), nullable=False)
    analysis = Column(Text, nullable=True)
    recommendation = Column(Text, nullable=True)
    improvement = Column(Text, nullable=True)
    
    # Relationship back to session
    session = relationship("AnalysisSession", back_populates="recommendations")

# Database connection setup
def get_database_connection():
    """Create a connection to the PostgreSQL database"""
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is not set")
    
    # Create engine
    engine = create_engine(database_url)
    
    # Create all tables if they don't exist
    Base.metadata.create_all(engine)
    
    # Create session factory
    Session = sessionmaker(bind=engine)
    
    return Session()

# Database operations
def save_analysis_session(session_name, motorcycle_info, video_filename, video_info, 
                       analysis_results, suspension_data, recommendations):
    """Save a complete analysis session to the database"""
    # Get database session
    db_session = get_database_connection()
    
    try:
        # Create a new analysis session
        new_session = AnalysisSession(
            session_name=session_name,
            motorcycle_info=motorcycle_info,
            video_filename=video_filename,
            video_duration=video_info.get('duration'),
            video_fps=video_info.get('fps')
        )
        
        # Add to session
        db_session.add(new_session)
        db_session.flush()  # To get the ID
        
        # Create metrics record
        metrics = AnalysisMetrics(
            session_id=new_session.id,
            avg_travel=analysis_results.get('avg_travel'),
            max_travel=analysis_results.get('max_travel'),
            min_travel=analysis_results.get('min_travel'),
            travel_range=analysis_results.get('travel_range'),
            max_compression_time=analysis_results.get('max_compression_time'),
            avg_compression_speed=analysis_results.get('avg_compression_speed'),
            avg_rebound_speed=analysis_results.get('avg_rebound_speed'),
            max_compression_speed=analysis_results.get('max_compression_speed'),
            max_rebound_speed=analysis_results.get('max_rebound_speed'),
            comp_rebound_ratio=analysis_results.get('comp_rebound_ratio'),
            oscillation_frequency=analysis_results.get('oscillation_frequency'),
            damping_ratio=analysis_results.get('damping_ratio')
        )
        db_session.add(metrics)
        
        # Add data points (limit to 1000 points to prevent database overload)
        sample_interval = max(1, len(suspension_data) // 1000)
        for i, data_point in enumerate(suspension_data):
            if i % sample_interval == 0:  # Save only sampled points
                dp = SuspensionDataPoint(
                    session_id=new_session.id,
                    time=data_point.get('time'),
                    distance=data_point.get('distance'),
                    velocity=data_point.get('velocity')
                )
                db_session.add(dp)
        
        # Add recommendations
        for component, rec in recommendations.items():
            recommendation = SuspensionRecommendation(
                session_id=new_session.id,
                component=component,
                analysis=rec.get('analysis'),
                recommendation=rec.get('recommendation'),
                improvement=rec.get('improvement')
            )
            db_session.add(recommendation)
        
        # Commit changes
        db_session.commit()
        
        return new_session.id
    
    except Exception as e:
        # Rollback in case of error
        db_session.rollback()
        raise e
    finally:
        # Close session
        db_session.close()

def get_analysis_sessions():
    """Retrieve all analysis sessions from the database"""
    # Get database session
    db_session = get_database_connection()
    
    try:
        # Query all sessions
        sessions = db_session.query(AnalysisSession)\
            .order_by(AnalysisSession.created_at.desc())\
            .all()
        
        # Return as list of dictionaries
        return [
            {
                'id': session.id,
                'session_name': session.session_name,
                'motorcycle_info': session.motorcycle_info,
                'created_at': session.created_at,
                'video_filename': session.video_filename
            }
            for session in sessions
        ]
    
    finally:
        # Close session
        db_session.close()

def get_analysis_session_by_id(session_id):
    """Retrieve a specific analysis session with all related data"""
    # Get database session
    db_session = get_database_connection()
    
    try:
        # Query the session
        session = db_session.query(AnalysisSession).filter(AnalysisSession.id == session_id).first()
        
        if not session:
            return None
        
        # Get metrics
        metrics = session.metrics
        metrics_dict = {}
        if metrics:
            # Convert metrics to dict excluding session_id and id
            metrics_dict = {
                column.name: getattr(metrics, column.name)
                for column in metrics.__table__.columns
                if column.name not in ('id', 'session_id')
            }
        
        # Get data points
        data_points = [
            {
                'time': dp.time,
                'distance': dp.distance,
                'velocity': dp.velocity
            }
            for dp in session.data_points
        ]
        
        # Get recommendations
        recommendations = {}
        for rec in session.recommendations:
            recommendations[rec.component] = {
                'analysis': rec.analysis,
                'recommendation': rec.recommendation,
                'improvement': rec.improvement
            }
        
        # Return complete session data
        return {
            'session': {
                'id': session.id,
                'session_name': session.session_name,
                'motorcycle_info': session.motorcycle_info,
                'created_at': session.created_at,
                'video_filename': session.video_filename,
                'video_duration': session.video_duration,
                'video_fps': session.video_fps
            },
            'metrics': metrics_dict,
            'data_points': data_points,
            'recommendations': recommendations
        }
    
    finally:
        # Close session
        db_session.close()

def delete_analysis_session(session_id):
    """Delete an analysis session and all related data"""
    # Get database session
    db_session = get_database_connection()
    
    try:
        # Query the session
        session = db_session.query(AnalysisSession).filter(AnalysisSession.id == session_id).first()
        
        if not session:
            return False
        
        # Delete the session (cascade will handle related records)
        db_session.delete(session)
        db_session.commit()
        
        return True
    
    except Exception as e:
        # Rollback in case of error
        db_session.rollback()
        raise e
    
    finally:
        # Close session
        db_session.close()
